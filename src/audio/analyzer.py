"""Noise level and speech-activity analysis (no speech recognition).

Every analysis window is described by its loudness (RMS and dBFS) and labelled
``silence``, ``background_noise`` or ``speech``. The label comes from signal
shape only -- no content is recognised or transcribed:

* **modulation** (required): speech energy rises and falls at syllable rate,
  whereas fans, hum and steady tones do not.
* **speech-band concentration**: most energy sits in roughly 100-3400 Hz.
* **tonality**: voiced speech is harmonic, i.e. spectrally far from flat.

A window is ``speech`` when it is above the noise floor, is modulated, and
satisfies at least one of the two spectral cues. These are transparent
heuristics, not a trained voice-activity detector; thresholds live in
``config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

import config
from src.audio.extractor import AudioSegment

ACTIVITY_SILENCE = "silence"
ACTIVITY_NOISE = "background_noise"
ACTIVITY_SPEECH = "speech"

_FRAME_SECONDS = 0.020
_MIN_DB = -100.0


@dataclass
class AudioWindowMetrics:
    start_time: float
    end_time: float
    rms_energy: float
    noise_level: float
    speech_activity: float
    is_silent: bool
    # Added fields keep defaults so older callers/reports stay valid.
    noise_db: float = _MIN_DB
    activity: str = ACTIVITY_SILENCE


def _band_ratio(power: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> float:
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(power[mask].sum() / total)


def _spectral_flatness(power: np.ndarray) -> float:
    """Geometric / arithmetic mean of the power spectrum (0 = tonal, ~0.56 = white)."""
    p = power + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def _energy_modulation(samples: np.ndarray, sample_rate: int) -> float:
    """Coefficient of variation of short-frame RMS across the window."""
    frame_len = int(_FRAME_SECONDS * sample_rate)
    if frame_len < 2 or len(samples) < 4 * frame_len:
        return 0.0
    usable = (len(samples) // frame_len) * frame_len
    frames = samples[:usable].reshape(-1, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    mean = float(frame_rms.mean())
    if mean <= 1e-9:
        return 0.0
    return float(frame_rms.std() / mean)


class AudioAnalyzer:
    def __init__(
        self,
        speech_threshold: float = config.SPEECH_ENERGY_THRESHOLD,
        noise_floor: float = config.NOISE_FLOOR_THRESHOLD,
    ) -> None:
        self.speech_threshold = speech_threshold
        self.noise_floor = noise_floor

    def classify_samples(self, samples: np.ndarray, sample_rate: int, rms: float) -> str:
        """Label one window as silence, background noise or speech."""
        if rms < self.noise_floor:
            return ACTIVITY_SILENCE
        if len(samples) < 8:
            return ACTIVITY_NOISE

        modulation = _energy_modulation(samples, sample_rate)
        if modulation < config.ENERGY_MODULATION_MIN:
            return ACTIVITY_NOISE

        windowed = (samples - samples.mean()) * np.hanning(len(samples))
        power = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
        band_ok = (
            _band_ratio(power, freqs, config.SPEECH_BAND_HZ) >= config.SPEECH_BAND_RATIO_MIN
        )
        tonal_ok = _spectral_flatness(power) <= config.SPECTRAL_FLATNESS_MAX
        return ACTIVITY_SPEECH if (band_ok or tonal_ok) else ACTIVITY_NOISE

    def analyze_segment(self, segment: AudioSegment) -> AudioWindowMetrics:
        samples = segment.samples
        rms = float(np.sqrt(np.mean(samples**2) + 1e-12))
        noise = min(1.0, rms / max(self.speech_threshold, 1e-6))
        noise_db = max(_MIN_DB, 20.0 * float(np.log10(rms)))
        activity = self.classify_samples(samples, segment.sample_rate, rms)

        speech = 0.0
        if activity == ACTIVITY_SPEECH:
            speech = min(1.0, (rms - self.noise_floor) / self.speech_threshold)
        return AudioWindowMetrics(
            start_time=segment.start_time,
            end_time=segment.end_time,
            rms_energy=round(rms, 6),
            noise_level=round(noise, 4),
            speech_activity=round(max(0.0, speech), 4),
            is_silent=activity == ACTIVITY_SILENCE,
            noise_db=round(noise_db, 2),
            activity=activity,
        )

    def analyze_all(self, segments: List[AudioSegment]) -> List[AudioWindowMetrics]:
        return [self.analyze_segment(s) for s in segments]


def summarize_audio(metrics: List[AudioWindowMetrics]) -> Dict[str, Any]:
    """Session-level audio context: speech/silence/noise shares and activity pattern.

    Returns ``{"available": False}`` when no audio was analysed so callers can
    tell "no audio source" apart from "audio was silent".
    """
    if not metrics:
        return {"available": False}

    durations = [max(m.end_time - m.start_time, 0.0) for m in metrics]
    total = sum(durations) or 1.0

    def share(label: str) -> float:
        return round(sum(d for m, d in zip(metrics, durations) if m.activity == label) / total, 4)

    speech_runs: List[float] = []
    silence_longest = 0.0
    current_speech = 0.0
    current_silence = 0.0
    transitions = 0
    previous_speech: bool | None = None
    for metric, duration in zip(metrics, durations):
        is_speech = metric.activity == ACTIVITY_SPEECH
        if previous_speech is not None and is_speech != previous_speech:
            transitions += 1
        previous_speech = is_speech
        if is_speech:
            current_speech += duration
            silence_longest = max(silence_longest, current_silence)
            current_silence = 0.0
        else:
            if current_speech:
                speech_runs.append(current_speech)
                current_speech = 0.0
            current_silence += duration if metric.activity == ACTIVITY_SILENCE else 0.0
            if metric.activity != ACTIVITY_SILENCE:
                silence_longest = max(silence_longest, current_silence)
                current_silence = 0.0
    if current_speech:
        speech_runs.append(current_speech)
    silence_longest = max(silence_longest, current_silence)

    return {
        "available": True,
        "duration_seconds": round(total, 2),
        "speech_ratio": share(ACTIVITY_SPEECH),
        "background_noise_ratio": share(ACTIVITY_NOISE),
        "silence_ratio": share(ACTIVITY_SILENCE),
        "mean_noise_db": round(float(np.mean([m.noise_db for m in metrics])), 2),
        "peak_noise_db": round(float(max(m.noise_db for m in metrics)), 2),
        "speech_bursts": len(speech_runs),
        "longest_speech_seconds": round(max(speech_runs), 2) if speech_runs else 0.0,
        "longest_silence_seconds": round(silence_longest, 2),
        "speech_transitions": transitions,
    }
