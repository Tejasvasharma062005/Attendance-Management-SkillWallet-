"""Capture microphone audio in memory for a live session."""

from __future__ import annotations

from queue import Empty, SimpleQueue

import numpy as np
import sounddevice as sd


class LiveAudioCapture:
    """Collect mono float audio from the default input device until stopped."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._chunks: SimpleQueue[np.ndarray] = SimpleQueue()
        self._statuses: SimpleQueue[str] = SimpleQueue()
        self._closed = False
        self._result: tuple[np.ndarray, list[str]] | None = None

        try:
            self._stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=self._record_audio,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            if hasattr(self, "_stream"):
                self._stream.close()
            raise RuntimeError(f"Could not start microphone capture: {exc}") from exc

    def _record_audio(self, indata, frames, _, status) -> None:
        if status:
            self._statuses.put(str(status))
        if frames:
            self._chunks.put(indata[:frames, 0].copy())

    def stop(self) -> tuple[np.ndarray, list[str]]:
        """Stop the input stream and return captured samples and device notices."""
        if self._result is not None:
            return self._result
        if not self._closed:
            self._closed = True
            try:
                self._stream.stop()
            except sd.PortAudioError as exc:
                raise RuntimeError(f"Could not stop microphone capture: {exc}") from exc
            finally:
                self._stream.close()

        chunks = []
        while True:
            try:
                chunks.append(self._chunks.get_nowait())
            except Empty:
                break

        statuses = []
        while True:
            try:
                statuses.append(self._statuses.get_nowait())
            except Empty:
                break

        waveform = (
            np.concatenate(chunks).astype(np.float32, copy=False)
            if chunks
            else np.empty(0, dtype=np.float32)
        )
        self._result = waveform, statuses
        return self._result

    def close(self) -> None:
        """Release the input device and discard any unprocessed samples."""
        self.stop()
