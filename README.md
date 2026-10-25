# SkillWallet Classroom Intelligence

A Streamlit platform for classroom-session attendance, visual engagement, audio
activity, teaching insights, and downloadable reports. When run locally,
processing and storage stay on that machine. When hosted, uploaded media is
processed by the hosting server; only use media you are authorized to upload.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer (Python 3.12 is the tested environment)
- A webcam for live sessions
- A working system microphone if microphone capture is selected
- Optional: `ffmpeg` on the PATH, used only to read the audio track inside
  `.mp4`/`.mov` files when the Python audio libraries cannot (otherwise supply a
  separate audio file)

## Install and start

From the project directory, create and activate a virtual environment in
PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_setup.py
python -m streamlit run app\streamlit_app.py
```

`verify_setup.py` checks the libraries, confirms the local vision models, and
runs a small pipeline self-test. The two MediaPipe models ship in `data/`; if
they are ever missing, run `python scripts\verify_setup.py --download-models`
once while online. After that the platform needs no internet.

Open the local URL printed by Streamlit (normally
`http://localhost:8501`). Keep the terminal running while using the platform.
To stop it, press `Ctrl+C` in that terminal.

On macOS or Linux, activate `.venv` with `source .venv/bin/activate` and run
`python -m streamlit run app/streamlit_app.py`.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Include `app/streamlit_app.py`,
   `requirements.txt`, `packages.txt`, and the two model files in `data/`.
   Do not commit classroom recordings or generated session reports.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) and choose
   **Create app**.
3. Select the repository and branch, set **Main file path** to
   `app/streamlit_app.py`, then deploy.
4. Wait for the build to finish and open the app URL provided by Streamlit.

`requirements.txt` installs the Python dependencies and `packages.txt` installs
the Linux system libraries needed by OpenCV and microphone support.

On Community Cloud, use **Recorded video** and upload media for analysis. The
live webcam and microphone controls access devices attached to the machine
running the Streamlit server, not a visitor's browser, so use those features
only when running the app locally on a computer with the appropriate devices.
Community Cloud's filesystem is not persistent across restarts or redeploys;
download reports you need to keep. Uploaded classroom media is sent to and
processed on the hosting server, so consider privacy, consent, and your
organization's data policies before deploying or uploading it.

## Use the platform

1. In **Start a new analysis**, choose **Live webcam** or **Recorded video**.
2. For a live session, choose no audio, the system microphone, or a separate
   audio file. Microphone permission and a working default input device are
   required when microphone capture is selected.
3. Set the capture duration and start the session. Live sessions show temporary
   participant IDs, attendance, visual signals, and an attention trend.
4. End the session to save its report. Recorded videos are analyzed after
   upload; optional separate audio is supported.
5. Browse **Dashboard**, **Teacher Central**, and **Analytics & Reports**. Select
   a saved report in the sidebar to revisit it, and download its PDF, its CSV
   data, or its timestamped attendance log.

For a command-line video analysis (optionally with a separate audio file):

```powershell
python main.py --video "C:\path\to\classroom.mp4"
python main.py --video "C:\path\to\classroom.mp4" --audio "C:\path\to\audio.wav"
```

For a command-line webcam session:

```powershell
python main.py --webcam --duration 60
```

Reports are stored locally under `data/sessions/<session-id>/report.json`.
Temporary uploaded media is removed after video processing or live-session
finalization. Do not put sensitive media in shared folders.

## How it works

| Stage | What it does | Code |
| --- | --- | --- |
| Face tracking | Detects faces (MediaPipe) and gives each a session-local ID, one ID per face per frame; IDs survive brief occlusion and returning to the same seat | `src/vision/face_tracker.py` |
| Attendance | Accumulates visibility per participant; present only after sustained presence, never from one frame. Records when presence was confirmed and writes a timestamped attendance log | `src/vision/attendance.py`, `src/storage/attendance_log.py` |
| Visual engagement | Head pose (face-landmark geometry), movement, inactivity | `src/vision/head_pose.py`, `engagement_visual.py` |
| Audio | Mono/22.05 kHz windows; loudness (dBFS) and a silence / background-noise / speech label from signal shape. No transcription | `src/audio/` |
| Fusion and scoring | Timestamp-aligned visual + audio score per 10 s interval, with a plain-language reason for every score | `src/agent/` |
| Session dynamics | Attention drops, interaction peaks, pacing decline | `src/agent/events.py` |
| Suggestions | Non-prescriptive teaching suggestions (instructor) and reflective timelines (one participant at a time) | `src/analytics/` |
| Reports | JSON report, PDF, CSV, attendance-log CSV, all local | `src/storage/` |

Rules worth knowing:

- **Attendance.** A participant counts as present after 30 s of accumulated
  visibility. For sessions shorter than 50 s the threshold scales to 60% of the
  session length so short clips can still be attended; the rule applied is
  stated in the report.
- **No audio, no penalty.** An interval with no audio (no source, or a dropout)
  is scored on visual signals alone. An interval with audio blends visual
  (60%) and speech activity (40%), so a silent room lowers the blended score.
  Weights, level thresholds, and event sensitivity are all in `config.py`.
- **Clock times.** Live sessions record real UTC times. A recorded video has no
  known recording time, so its attendance log uses positions in the video.

## Privacy and interpretation

- Video, microphone input, and reports are processed or stored locally; this
  project does not send classroom media to an external service when run locally.
  In a hosted deployment, uploaded media is processed on the hosting server.
- Live microphone samples are buffered in memory for analysis, not retained as
  an audio recording. Audio from a video container is decoded in memory too.
  Analysis uses signal-shape features (loudness, modulation, spectrum); it does
  not transcribe speech. Reports store only per-window numbers and labels.
- The platform does not perform emotion recognition, assign grades, or rank
  learners. Participant IDs are temporary and specific to each session.
- Head pose, movement, inactivity, and audio activity are approximate signals,
  not reliable measures of an individual learner's attention or performance.
  Teaching suggestions are supportive prompts, not evaluations.
- When the app is run on another computer or hosted remotely, webcam and
  microphone capture refer to devices available to that computer/server, not
  automatically to the browser user's devices. Run the app locally for access
  to your own computer's camera and microphone.

## Test

Run the automated suite with:

```powershell
python -m pytest tests -v
```

The suite covers tracking, attendance, audio classification, fusion, scoring,
events, suggestions, exports, and the saved report. Tests mock camera and
microphone hardware. To verify real devices, start a live
session from the dashboard and confirm the camera preview and, if selected,
audio metrics before ending the session. Live device access depends on operating
system permissions and the selected input device.

## Known limits

- Speech detection is a transparent heuristic (modulation plus spectral cues),
  checked on synthetic signals and not a trained voice-activity detector. Music
  or a sustained sung note can look like speech. Tune `config.py` for your room.
- Presence time is accumulated per tracked face. A different person who sits in
  a vacated seat inherits that seat's ID.
- Head pose is estimated from facial landmarks and is a rough attention proxy.
