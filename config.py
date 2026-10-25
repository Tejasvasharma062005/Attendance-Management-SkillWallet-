"""Project-wide configuration for the Multimodal Attendance & Engagement Agent."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Vision / attendance
# A participant is marked present once their accumulated visibility reaches
# the threshold. For sessions shorter than PRESENCE_THRESHOLD_SECONDS / fraction
# the threshold scales down (see PresenceAttendanceEngine.effective_threshold),
# otherwise a 20-second clip could never mark anyone present.
PRESENCE_THRESHOLD_SECONDS = 30.0
PRESENCE_MAX_SESSION_FRACTION = 0.6
TRACKING_MAX_DISTANCE = 80.0
MIN_FACE_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5

# Engagement (visual)
INACTIVITY_SECONDS = 8.0
MOVEMENT_THRESHOLD = 0.015

# Audio
AUDIO_SAMPLE_RATE = 22050
AUDIO_WINDOW_SECONDS = 1.0
SPEECH_ENERGY_THRESHOLD = 0.02
NOISE_FLOOR_THRESHOLD = 0.005
# Speech-vs-noise discrimination (energy alone cannot tell a fan from a voice).
# Human speech concentrates energy in roughly 300-3400 Hz and is far from
# spectrally flat; steady background noise is the opposite.
SPEECH_BAND_HZ = (100.0, 3400.0)
SPEECH_BAND_RATIO_MIN = 0.45
SPECTRAL_FLATNESS_MAX = 0.35
# Speech energy rises and falls at syllable rate; steady noise does not.
# Measured as std/mean of 20 ms frame RMS inside each analysis window.
ENERGY_MODULATION_MIN = 0.35

# Scoring intervals
ENGAGEMENT_INTERVAL_SECONDS = 10.0

# Fusion weights (must sum to 1.0)
# When an interval has no audio at all (no audio source, or an input dropout),
# the visual weight is renormalised to 1.0 for that interval instead of letting
# the missing audio drag the score down.
WEIGHT_VISUAL = 0.6
WEIGHT_AUDIO = 0.4

# Engagement level thresholds (transparent and adjustable)
ENGAGEMENT_HIGH_THRESHOLD = 0.65
ENGAGEMENT_MEDIUM_THRESHOLD = 0.35

# Session-dynamics event detection
ATTENTION_DROP_DELTA = 0.20          # score fall between consecutive intervals
INTERACTION_PEAK_SPEECH = 0.60       # interval speech activity counted as a peak
PACING_DECLINE_DELTA = 0.15          # last-third vs first-third engagement fall
PACING_MIN_INTERVALS = 6             # need this many intervals to judge pacing

# Live (in-session) recommendations
LIVE_LOW_ATTENTION = 0.45
LIVE_LOW_ATTENTION_SECONDS = 20.0
