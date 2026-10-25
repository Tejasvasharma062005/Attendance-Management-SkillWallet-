# Changes

Gap analysis against SkillWallet.pdf (Epics 1-6). Defects fixed, then gaps filled.

## Defects fixed
- **Engagement was capped at 60% without audio.** Missing audio counted as zero
  speech, so `combined = 0.6 x visual` and "high" engagement was unreachable.
  Intervals with no audio are now scored on visual signals (`src/agent/fusion.py`).
- **Short sessions could never mark anyone present.** The fixed 30 s rule meant a
  participant visible for a whole 13 s session was "absent" (this was in the two
  saved reports). The threshold now scales with session length.
- **Tracker could give two faces the same ID.** Assignment is now one-to-one per frame.
- **Head pose:** landmark results could be reused across faces, and a face the
  landmarker missed was scored as a perfect 1.0. Matching is now one-to-one with
  a bounding-box fallback.
- **PDF export:** table headers were dark-on-dark, and markup in the suggestions
  section printed as literal `<b>`/`<font>` text.
- **Focus highlights** claimed a window from the first to the last strong sample,
  even across long gaps; they now use the longest contiguous stretch.

## Gaps filled
- Timestamped attendance log (first seen, marked present, last seen): saved in the
  report, shown in the dashboard, downloadable as CSV, included in PDF/CSV.
- Audio: dBFS level, silence / background-noise / speech labels, session audio
  summary, raw windows saved; ffmpeg fallback to read audio inside video files.
- Explainable scoring: every interval has a plain-language rationale; level
  thresholds moved to `config.py`.
- Session dynamics: attention drops, interaction peaks, pacing decline.
- Teaching suggestions use those events and tell speech noise from other noise.
- Per-participant focus timeline (one participant at a time, no ranking).
- Live dashboard: trend-based group suggestions; dynamic attendance rule text.
- CLI `--audio`; `verify_setup.py` checks models and runs a pipeline self-test;
  `requirements.txt` grouped; README documents the pipeline and its limits.

## Tests
63 original tests kept (one fixture changed: it fed a DC offset as "speech").
65 new tests cover each change above.

## Not done on purpose
The PDF's screenshots show a React + FastAPI + Groq (cloud LLM) stack with emotion
labels and a leaderboard. That conflicts with the PDF's own text (offline,
privacy-preserving, rule-based, no ranking or evaluation), so it was not built.
