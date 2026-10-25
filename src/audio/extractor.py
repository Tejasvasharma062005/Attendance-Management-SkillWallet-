"""Extract and window audio from files or video sources."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np

import config


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    samples: np.ndarray
    sample_rate: int


class AudioExtractor:
    """Load audio and split into fixed-duration windows aligned to session time."""

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        window_seconds: float = config.AUDIO_WINDOW_SECONDS,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds

    def _decode_with_ffmpeg(self, path: Path) -> Optional[tuple[np.ndarray, int]]:
        """Decode the audio track to mono PCM through ffmpeg, entirely in memory.

        Used when librosa cannot open a container such as .mp4 (it depends on
        optional decoder back-ends). Samples are piped back over stdout, so no
        temporary audio file is written. Returns None when ffmpeg is missing,
        the file has no audio stream, or decoding fails.
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return None
        try:
            result = subprocess.run(
                [
                    ffmpeg, "-v", "error", "-i", str(path), "-vn",
                    "-ac", "1", "-ar", str(self.sample_rate),
                    "-f", "s16le", "-",
                ],
                capture_output=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0 or len(result.stdout) < 2:
            return None
        pcm = np.frombuffer(result.stdout[: len(result.stdout) // 2 * 2], dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0, self.sample_rate

    def from_file(self, path: str | Path) -> tuple[np.ndarray, int]:
        try:
            y, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
            return y, sr
        except Exception:
            decoded = self._decode_with_ffmpeg(Path(path))
            if decoded is None:
                raise
            return decoded

    def from_video(self, video_path: str | Path) -> Optional[tuple[np.ndarray, int]]:
        path = Path(video_path)
        if not path.exists():
            return None
        try:
            y, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
            if len(y):
                return y, sr
        except Exception:
            pass
        return self._decode_with_ffmpeg(path)

    def to_segments(self, waveform: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        window_len = int(self.window_seconds * sample_rate)
        segments: List[AudioSegment] = []
        if window_len <= 0:
            return segments
        for i in range(0, len(waveform), window_len):
            chunk = waveform[i : i + window_len]
            if len(chunk) == 0:
                continue
            start = i / sample_rate
            end = start + len(chunk) / sample_rate
            segments.append(
                AudioSegment(
                    start_time=start,
                    end_time=end,
                    samples=chunk.astype(np.float32),
                    sample_rate=sample_rate,
                )
            )
        return segments
