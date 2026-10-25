"""PDF-aligned Streamlit dashboard for local attendance and engagement reports."""

from __future__ import annotations

import cv2
import re
import sys
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.analytics.live_recommendations import generate_live_recommendations
from src.pipeline.live_session import LiveWebcamSession
from src.pipeline.session_processor import SessionProcessor
from src.storage.attendance_log import attendance_log_to_csv, build_attendance_log
from src.storage.logs import load_session_report
from src.storage.report_exports import build_session_csv, build_session_pdf


_PAGES = ("Dashboard", "Teacher Central", "Analytics & Reports")


@st.cache_data(ttl=5, show_spinner=False)
def _load_reports() -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    for directory in sorted(config.SESSIONS_DIR.iterdir(), reverse=True):
        if directory.is_dir() and (directory / "report.json").is_file():
            report = load_session_report(directory.name)
            if report is not None:
                reports[directory.name] = report
    return reports


def _analysis(report: Dict[str, Any]) -> Dict[str, Any]:
    return report["payload"]["analysis"]


def _average(records: List[Dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _full_width_kwargs(widget: Any) -> Dict[str, Any]:
    if "width" in signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def _style_chart(figure, y_range: tuple[float, float] = (0, 1)):
    figure.update_layout(
        height=320,
        margin=dict(l=12, r=12, t=44, b=12),
        paper_bgcolor="#101a2c",
        plot_bgcolor="#101a2c",
        font=dict(color="#e6edf7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor="#29364a", zerolinecolor="#29364a"),
        yaxis=dict(
            gridcolor="#29364a",
            zerolinecolor="#29364a",
            range=list(y_range),
        ),
    )
    return figure


def _finish_live_session(session: LiveWebcamSession) -> None:
    try:
        report = session.finalize()
    finally:
        _cleanup_staged_live_audio()
    st.session_state.pop("live_session", None)
    st.session_state.pop("live_signal_history", None)
    st.session_state["active_session_id"] = report.session_id
    st.session_state["pending_session_selection"] = report.session_id
    st.session_state["analysis_notice"] = (
        f"Session saved locally. Report: {report.report_path}"
    )
    _load_reports.clear()


def _remove_staged_upload(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        st.warning(f"Could not remove temporary upload {path}: {exc}")


def _cleanup_staged_live_audio() -> None:
    audio_path = st.session_state.pop("staged_live_audio_path", None)
    if audio_path is not None:
        _remove_staged_upload(Path(audio_path))


@st.cache_data(show_spinner=False)
def _cached_pdf_report(session_id: str, payload: Dict[str, Any]) -> bytes:
    return build_session_pdf(session_id, payload)


@st.cache_data(show_spinner=False)
def _cached_csv_report(session_id: str, payload: Dict[str, Any]) -> str:
    return build_session_csv(session_id, payload)


def _attendance_events(session_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stored attendance log, or one rebuilt from records for older reports."""
    events = payload.get("attendance_log")
    if events is None:
        events = build_attendance_log(
            payload["analysis"].get("attendance", []),
            session_id,
            payload.get("session_started_at"),
        )
    return events


@st.cache_data(show_spinner=False)
def _cached_attendance_log_csv(session_id: str, payload: Dict[str, Any]) -> str:
    return attendance_log_to_csv(_attendance_events(session_id, payload))


def _render_export_buttons(
    session_id: str, payload: Dict[str, Any], key_prefix: str
) -> None:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id).strip("._") or "session"
    pdf_data = _cached_pdf_report(session_id, payload)
    csv_data = _cached_csv_report(session_id, payload)
    log_data = _cached_attendance_log_csv(session_id, payload)
    pdf_column, csv_column, log_column = st.columns(3)
    with pdf_column:
        st.download_button(
            "Download PDF report",
            data=pdf_data,
            file_name=f"skillwallet-{safe_id}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
            **_full_width_kwargs(st.download_button),
        )
    with csv_column:
        st.download_button(
            "Download CSV data",
            data=csv_data,
            file_name=f"skillwallet-{safe_id}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
            **_full_width_kwargs(st.download_button),
        )
    with log_column:
        st.download_button(
            "Download attendance log",
            data=log_data,
            file_name=f"skillwallet-{safe_id}-attendance-log.csv",
            mime="text/csv",
            key=f"{key_prefix}_attendance_log",
            **_full_width_kwargs(st.download_button),
        )


def _render_attendance(
    analysis: Dict[str, Any],
    payload: Dict[str, Any] | None = None,
    session_id: str = "",
) -> None:
    attendance = analysis.get("attendance", [])
    st.subheader("Attendance overview")
    if attendance:
        attendance_frame = pd.DataFrame(attendance).drop(
            columns=["threshold_seconds"], errors="ignore"
        ).rename(
            columns={
                "track_id": "Participant ID",
                "total_visible_seconds": "Visible time (s)",
                "present": "Attendance threshold met",
                "first_seen": "First seen (s)",
                "last_seen": "Last seen (s)",
                "marked_present_at": "Marked present at (s)",
            }
        )
        st.dataframe(
            attendance_frame,
            hide_index=True,
            **_full_width_kwargs(st.dataframe),
        )
        threshold = (payload or {}).get("attendance_threshold_seconds")
        if threshold is not None:
            st.caption(
                f"Presence rule: counted present after {float(threshold):g}s of "
                "accumulated visibility, never from a single frame. For sessions "
                "shorter than the standard 30s the rule scales with session length."
            )
    else:
        st.info("No participants were detected in this session.")

    if payload is not None and attendance:
        events = _attendance_events(session_id, payload)
        with st.expander("Timestamped attendance log"):
            if events:
                log_frame = pd.DataFrame(events).rename(
                    columns={
                        "participant": "Participant",
                        "event": "Event",
                        "session_time": "Session time",
                        "timestamp_utc": "Clock time (UTC)",
                        "details": "Details",
                    }
                )
                st.dataframe(
                    log_frame[
                        ["Participant", "Event", "Session time", "Clock time (UTC)", "Details"]
                    ],
                    hide_index=True,
                    **_full_width_kwargs(st.dataframe),
                )
            else:
                st.info("No attendance events were recorded.")
            st.caption(
                "Clock times appear for live sessions only. A recorded video has no "
                "known recording time, so it is logged by position in the video."
            )


def _render_dashboard(session_id: str, report: Dict[str, Any]) -> None:
    payload = report["payload"]
    analysis = _analysis(report)
    attendance = analysis.get("attendance", [])
    intervals = analysis.get("fused_intervals", [])
    duration = float(analysis.get("session_duration_seconds", 0.0))
    present_count = sum(bool(person.get("present")) for person in attendance)
    avg_attention = _average(intervals, "visual_score")
    audio_intervals = [
        interval
        for interval in intervals
        if interval.get("metadata", {}).get("sample_count_audio", 0) > 0
    ]
    avg_audio = _average(audio_intervals, "audio_score")
    source = "Recorded video" if payload.get("video") else payload.get("source", "Unknown")

    st.title("Session Dashboard")
    st.caption("Classroom intelligence overview · privacy-conscious, local analysis")
    status, session, source_column = st.columns([1.2, 2, 1.2])
    status.success("Analysis complete")
    session.caption(f"Session · {session_id}")
    source_column.caption(f"Input · {source}")

    duration_column, participants_column, present_column, attention_column = st.columns(4)
    duration_column.metric("Session duration", f"{duration / 60:.1f} min")
    participants_column.metric("Participants detected", len(attendance))
    present_column.metric("Attendance threshold met", present_count)
    attention_column.metric(
        "Average visual attention",
        f"{avg_attention * 100:.0f}%" if avg_attention is not None else "N/A",
    )

    video_path = payload.get("video")
    preview_column, context_column = st.columns([1.8, 1])
    with preview_column:
        st.subheader("Session recording")
        if video_path and Path(video_path).is_file():
            st.video(Path(video_path))
        elif payload.get("source") == "webcam":
            st.info(
                "This report was analyzed from a webcam session. "
                "A recording is not retained by the local pipeline."
            )
        else:
            st.info("The original recording is not available for preview.")
    with context_column:
        st.subheader("Audio context")
        audio_summary = payload.get("audio_summary") or {}
        if audio_summary.get("available"):
            st.metric(
                "Speech activity",
                f"{float(audio_summary.get('speech_ratio', 0.0)) * 100:.0f}% of session",
            )
            st.metric(
                "Mean noise level",
                f"{float(audio_summary.get('mean_noise_db', 0.0)):.0f} dBFS",
            )
            st.caption(
                f"Silence {float(audio_summary.get('silence_ratio', 0.0)) * 100:.0f}% · "
                f"background noise "
                f"{float(audio_summary.get('background_noise_ratio', 0.0)) * 100:.0f}% · "
                f"{int(audio_summary.get('speech_bursts', 0))} speech burst(s). "
                "Signal shape only; speech is not transcribed."
            )
        elif avg_audio is None:
            st.metric("Speech activity proxy", "Not available")
            st.caption("No audio segments were available for this session.")
        else:
            noise_values = [
                float(interval.get("metadata", {}).get("avg_noise", 0.0))
                for interval in audio_intervals
            ]
            noise_score = sum(noise_values) / len(noise_values) if noise_values else 0.0
            st.metric("Speech activity proxy", f"{avg_audio * 100:.0f}%")
            st.metric("Ambient noise proxy", f"{noise_score * 100:.0f}%")
            st.caption("Audio features only; speech is not transcribed.")
        for warning in payload.get("audio_capture_warnings", []):
            st.warning(f"Microphone capture notice: {warning}")

        st.subheader("AI insights")
        suggestions = payload.get("teaching_suggestions", [])
        if suggestions:
            for suggestion in suggestions[:2]:
                with st.container():
                    st.markdown(f"**{suggestion['category'].replace('_', ' ').title()}**")
                    st.write(suggestion["message"])
                    st.caption(
                        f"{suggestion['time_range']} · {suggestion['evidence']}"
                    )
        else:
            st.info("No teaching suggestions were generated for this session.")

    _render_attendance(analysis, payload, session_id)

    timeline = analysis.get("engagement_timeline", [])
    st.subheader("Engagement trend")
    if timeline:
        trend = pd.DataFrame(timeline)
        trend["time"] = (trend["start_time"] + trend["end_time"]) / 2
        figure = px.line(
            trend,
            x="time",
            y="score",
            color="level",
            markers=True,
            title="Engagement by session interval",
            color_discrete_map={"high": "#36d399", "medium": "#f4c95d", "low": "#fb7185"},
        )
        st.plotly_chart(
            _style_chart(figure),
            key="dashboard_engagement_trend",
            **_full_width_kwargs(st.plotly_chart),
        )
    else:
        st.info("No engagement intervals are available for this report.")

    _render_session_events(analysis)


_EVENT_LABELS = {
    "attention_drop": "Attention drop",
    "interaction_peak": "Interaction peak",
    "pacing_decline": "Pacing decline",
}


def _render_session_events(analysis: Dict[str, Any]) -> None:
    """List attention drops, interaction peaks and pacing issues on the timeline."""
    events = analysis.get("events", [])
    st.subheader("Session dynamics")
    if not events:
        st.info(
            "No sharp attention drops, interaction peaks or pacing decline were "
            "detected in this session."
        )
        return
    for event in events:
        label = _EVENT_LABELS.get(event.get("event_type", ""), "Event")
        st.markdown(
            f"**{label}** · {float(event.get('start_time', 0)):.0f}s to "
            f"{float(event.get('end_time', 0)):.0f}s"
        )
        st.caption(event.get("detail", ""))


def _history_summary(reports: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for session_id, report in reports.items():
        analysis = _analysis(report)
        intervals = analysis.get("fused_intervals", [])
        attendance = analysis.get("attendance", [])
        average_engagement = _average(intervals, "combined_score")
        rows.append(
            {
                "Session": session_id,
                "Duration (min)": round(
                    float(analysis.get("session_duration_seconds", 0.0)) / 60, 1
                ),
                "Average engagement": (
                    round(average_engagement * 100, 1)
                    if average_engagement is not None
                    else None
                ),
                "Participants": len(attendance),
                "Attendance threshold met": sum(
                    bool(record.get("present")) for record in attendance
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_teacher_central(
    session_id: str,
    report: Dict[str, Any],
    reports: Dict[str, Dict[str, Any]],
) -> None:
    payload = report["payload"]
    analysis = _analysis(report)
    intervals = analysis.get("fused_intervals", [])
    attendance = analysis.get("attendance", [])
    suggestions = payload.get("teaching_suggestions", [])
    avg_engagement = _average(intervals, "combined_score")
    head_pose_values = [
        float(interval["metadata"]["head_pose_score"])
        for interval in intervals
        if interval.get("metadata", {}).get("head_pose_score") is not None
    ]
    avg_head_pose = (
        sum(head_pose_values) / len(head_pose_values) if head_pose_values else None
    )

    st.title("Teacher Profile")
    st.caption(
        "Session insights for reflective teaching · not a teacher rating"
    )
    engagement_column, attendance_column, duration_column = st.columns(3)
    engagement_column.metric(
        "Average session engagement",
        f"{avg_engagement * 100:.1f}/100" if avg_engagement is not None else "N/A",
    )
    attendance_column.metric(
        "Participants meeting attendance threshold",
        f"{sum(bool(person.get('present')) for person in attendance)} / {len(attendance)}",
    )
    duration_column.metric(
        "Session duration",
        f"{float(analysis.get('session_duration_seconds', 0.0)) / 60:.1f} min",
    )

    st.subheader("Session reflection")
    strengths_column, recommendations_column = st.columns(2)
    high_intervals = [point for point in analysis.get("engagement_timeline", []) if point["level"] == "high"]
    with strengths_column:
        st.markdown("#### Observed strengths")
        if high_intervals:
            peak = max(high_intervals, key=lambda point: point["score"])
            st.write(
                f"Engagement was strongest around "
                f"{peak['start_time']:.0f}-{peak['end_time']:.0f} seconds."
            )
        elif avg_head_pose is not None:
            st.write(
                f"Average forward-facing head-pose signal was "
                f"{avg_head_pose * 100:.0f}%."
            )
        else:
            st.write("More visual signal data is needed to identify session strengths.")
    with recommendations_column:
        st.markdown("#### Recommendations")
        if suggestions:
            for suggestion in suggestions:
                st.write(
                    f"**{suggestion['category'].replace('_', ' ').title()}** "
                    f"({suggestion['time_range']}): {suggestion['message']}"
                )
        else:
            st.write("No recommendations were generated for this session.")

    st.subheader("Session history")
    history = _history_summary(reports)
    if not history.empty:
        st.dataframe(history, hide_index=True, **_full_width_kwargs(st.dataframe))
        history_chart = history.dropna(subset=["Average engagement"])
        if not history_chart.empty:
            figure = px.line(
                history_chart,
                x="Session",
                y="Average engagement",
                markers=True,
                title="Session-level engagement trend",
            )
            figure.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(
                _style_chart(figure, y_range=(0, 100)),
                key=f"teacher_history_{session_id}",
                **_full_width_kwargs(st.plotly_chart),
            )
    else:
        st.info("Saved sessions will appear here for session-to-session reflection.")


def _render_analytics(session_id: str, report: Dict[str, Any]) -> None:
    payload = report["payload"]
    analysis = _analysis(report)
    intervals = analysis.get("fused_intervals", [])

    st.title("Analytics & Reports")
    st.caption("Explore interval-level engagement and download the local session report.")
    _render_export_buttons(session_id, payload, key_prefix="analytics")

    if intervals:
        timeline = analysis.get("engagement_timeline", [])
        frame = pd.DataFrame(
            [
                {
                    "Time": (interval["start_time"] + interval["end_time"]) / 2,
                    "Engagement": interval["combined_score"],
                    "Engagement level": timeline[index].get("level", "")
                    if index < len(timeline)
                    else "",
                    "Visual attention": interval["visual_score"],
                    "Head pose": interval.get("metadata", {}).get("head_pose_score"),
                    "Speech activity": (
                        interval["audio_score"]
                        if interval.get("metadata", {}).get("sample_count_audio", 0) > 0
                        else None
                    ),
                    "Noise level": (
                        interval.get("metadata", {}).get("avg_noise")
                        if interval.get("metadata", {}).get("sample_count_audio", 0) > 0
                        else None
                    ),
                }
                for index, interval in enumerate(intervals)
            ]
        )
        engagement_column, visual_column = st.columns(2)
        with engagement_column:
            st.subheader("Engagement trend")
            figure = px.line(
                frame,
                x="Time",
                y="Engagement",
                markers=True,
                title="Multimodal engagement by interval",
            )
            st.plotly_chart(
                _style_chart(figure),
                key=f"analytics_engagement_{session_id}",
                **_full_width_kwargs(st.plotly_chart),
            )
        with visual_column:
            st.subheader("Visual attention over time")
            visual_frame = frame.melt(
                id_vars=["Time"],
                value_vars=["Visual attention", "Head pose"],
                var_name="Signal",
                value_name="Score",
            ).dropna(subset=["Score"])
            figure = px.line(
                visual_frame,
                x="Time",
                y="Score",
                color="Signal",
                markers=True,
                title="Visual attention and forward-facing head pose",
            )
            st.plotly_chart(
                _style_chart(figure),
                key=f"analytics_visual_{session_id}",
                **_full_width_kwargs(st.plotly_chart),
            )

        st.subheader("Engagement levels over time")
        figure = px.bar(
            frame,
            x="Time",
            y="Engagement",
            color="Engagement level",
            title="Interval engagement distribution",
            color_discrete_map={
                "high": "#36d399",
                "medium": "#f4c95d",
                "low": "#fb7185",
            },
        )
        st.plotly_chart(
            _style_chart(figure),
            key=f"analytics_levels_{session_id}",
            **_full_width_kwargs(st.plotly_chart),
        )

        rationale_rows = [
            {
                "Interval": f"{point['start_time']:.0f}s to {point['end_time']:.0f}s",
                "Score": point["score"],
                "Level": point["level"],
                "Why this score": point.get("rationale") or "Not recorded for this report.",
            }
            for point in timeline
        ]
        with st.expander("Why each interval scored the way it did"):
            st.dataframe(
                pd.DataFrame(rationale_rows),
                hide_index=True,
                **_full_width_kwargs(st.dataframe),
            )
            st.caption(
                "Scores are rule-based: weighted visual attention (head pose, movement, "
                "inactivity) plus speech activity when audio exists. Without audio an "
                "interval is scored on visual signals alone."
            )

        audio_frame = frame.dropna(subset=["Speech activity", "Noise level"], how="all")
        st.subheader("Audio context over time")
        if not audio_frame.empty:
            audio_long = audio_frame.melt(
                id_vars=["Time"],
                value_vars=["Speech activity", "Noise level"],
                var_name="Signal",
                value_name="Score",
            ).dropna(subset=["Score"])
            figure = px.line(
                audio_long,
                x="Time",
                y="Score",
                color="Signal",
                markers=True,
                title="Speech activity and ambient noise proxies",
            )
            st.plotly_chart(
                _style_chart(figure),
                key=f"analytics_audio_{session_id}",
                **_full_width_kwargs(st.plotly_chart),
            )
            st.caption("Speech activity is an energy-based proxy; no speech is transcribed.")
        else:
            st.info("No audio input was available for this session.")
    else:
        st.info("No interval analytics are available for this report.")

    _render_session_events(analysis)
    _render_attendance(analysis, payload, session_id)
    student_analytics = payload.get("student_analytics", [])
    st.subheader("Participant reflections")
    if student_analytics:
        reflections = pd.DataFrame(student_analytics).drop(
            columns=["timeline"], errors="ignore"
        ).rename(
            columns={
                "track_id": "Participant ID",
                "attended": "Attendance threshold met",
                "visible_seconds": "Visible time (s)",
                "avg_attention": "Average attention proxy",
                "engagement_consistency": "Engagement consistency",
                "focus_highlights": "Focus highlights",
                "reflection_note": "Reflection",
            }
        )
        st.dataframe(reflections, hide_index=True, **_full_width_kwargs(st.dataframe))
        _render_participant_timeline(session_id, student_analytics)
    else:
        st.info("No participant reflections are available.")


def _render_participant_timeline(
    session_id: str, student_analytics: List[Dict[str, Any]]
) -> None:
    """One participant's own focus timeline (never overlaid with others)."""
    with_timeline = [item for item in student_analytics if item.get("timeline")]
    st.subheader("Individual focus timeline")
    if not with_timeline:
        st.info("Per-participant timelines are not available for this report.")
        return
    labels = {f"Participant {item['track_id']}": item for item in with_timeline}
    choice = st.selectbox(
        "Participant",
        list(labels),
        key=f"participant_timeline_{session_id}",
    )
    item = labels[choice]
    frame = pd.DataFrame(item["timeline"])
    frame["Time (s)"] = (frame["start_time"] + frame["end_time"]) / 2
    frame = frame.rename(columns={"attention": "Visual attention"})
    figure = px.line(
        frame,
        x="Time (s)",
        y="Visual attention",
        markers=True,
        title=f"{choice}: visual attention by interval",
    )
    st.plotly_chart(
        _style_chart(figure),
        key=f"participant_timeline_chart_{session_id}_{item['track_id']}",
        **_full_width_kwargs(st.plotly_chart),
    )
    st.caption(
        f"{item.get('focus_highlights', '')} Shown for one participant at a time: "
        "this view is for self-reflection, not comparison or ranking."
    )


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remaining:02d}"


@st.fragment(run_every=0.5)
def _render_live_session(session: LiveWebcamSession) -> None:
    if st.session_state.get("live_session") is not session:
        return

    live_frame = None
    try:
        live_frame = session.step()
    except RuntimeError as exc:
        session.close()
        _cleanup_staged_live_audio()
        st.session_state.pop("live_session", None)
        st.error(f"Live webcam session stopped: {exc}")
        return

    if live_frame is not None:
        frame = live_frame.frame_bgr.copy()
        scores_by_track = {sample.track_id: sample for sample in live_frame.samples}
        for face in live_frame.faces:
            x, y, width, height = face.bbox
            sample = scores_by_track.get(face.track_id)
            score_text = (
                f"Attention {sample.attention_proxy:.0%}"
                if sample is not None
                else "Attention n/a"
            )
            cv2.rectangle(frame, (x, y), (x + width, y + height), (54, 211, 153), 2)
            cv2.putText(
                frame,
                f"ID {face.track_id} | {score_text}",
                (x, max(y - 10, 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (54, 211, 153),
                2,
                cv2.LINE_AA,
            )

        if live_frame.samples:
            history = st.session_state.setdefault("live_signal_history", [])
            history.append(
                {
                    "Time (s)": live_frame.elapsed_seconds,
                    "Visual attention": sum(
                        sample.attention_proxy for sample in live_frame.samples
                    )
                    / len(live_frame.samples),
                    "Head pose": sum(
                        sample.head_pose_score for sample in live_frame.samples
                    )
                    / len(live_frame.samples),
                }
            )
            del history[:-120]

        preview_column, status_column = st.columns([1.8, 1])
        with preview_column:
            st.markdown("#### Live classroom view")
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="Live local analysis · face boxes use temporary session IDs",
                **_full_width_kwargs(st.image),
            )
        with status_column:
            records = session.current_attendance
            present_count = sum(record.present for record in records)
            attention = session.current_attention
            st.markdown("#### Engagement live")
            st.metric(
                "Visual attention proxy",
                f"{attention:.0%}" if attention is not None else "Waiting for a detected face",
            )
            st.metric("Students present", present_count)
            st.metric("Currently visible", len(live_frame.faces))
            st.metric(
                "Session time",
                f"{_format_elapsed(session.elapsed_seconds)} / "
                f"{_format_elapsed(session.duration_seconds)}",
            )
            st.caption(
                f"Attendance is confirmed after "
                f"{session.attendance_threshold_seconds:g} seconds of visibility."
            )

        signals = st.session_state.get("live_signal_history", [])
        if signals:
            st.markdown("#### Attention trend")
            trend = pd.DataFrame(signals)
            figure = px.line(
                trend,
                x="Time (s)",
                y=["Visual attention", "Head pose"],
                markers=True,
                title="Live visual signals",
                color_discrete_map={
                    "Visual attention": "#46d69b",
                    "Head pose": "#8b83f6",
                },
            )
            st.plotly_chart(
                _style_chart(figure),
                key="live_attention_trend",
                **_full_width_kwargs(st.plotly_chart),
            )

        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown("#### Participants in this session")
            if records:
                roster = pd.DataFrame(
                    [
                        {
                            "Participant ID": f"Participant {record.track_id}",
                            "Visible time": f"{record.total_visible_seconds:.1f}s",
                            "Attendance status": (
                                "Present" if record.present else "Being tracked"
                            ),
                        }
                        for record in records
                    ]
                )
                st.dataframe(
                    roster,
                    hide_index=True,
                    **_full_width_kwargs(st.dataframe),
                )
            else:
                st.info("Participants appear here as they are detected.")
        with right_column:
            st.markdown("#### Audio context")
            if session.capture_microphone:
                st.info(
                    "Microphone capture is active. Audio stays in memory and is "
                    "analyzed locally when the session ends; speech is not transcribed."
                )
            elif session.audio_path is not None:
                st.info(
                    "Optional audio is queued for local processing and will be "
                    "aligned to the start of this session."
                )
            else:
                st.info(
                    "Audio is optional. Attach a local audio file before starting "
                    "to include speech-activity and ambient-noise proxies."
                )
            st.markdown("#### Live suggestions")
            recent_signals = [
                {"time": row["Time (s)"], "attention": row["Visual attention"]}
                for row in st.session_state.get("live_signal_history", [])
            ]
            for rec in generate_live_recommendations(
                recent_signals, live_frame.elapsed_seconds, len(live_frame.faces)
            ):
                render = {
                    "success": st.success,
                    "warning": st.warning,
                }.get(rec.kind, st.info)
                render(rec.message)
                st.caption(rec.evidence)
            st.caption(
                "Group-level, trend-based cues for the instructor, not learner evaluations."
            )
    elif not session.finished:
        st.info("Waiting for the next camera frame…")

    if session.finished:
        try:
            _finish_live_session(session)
        except (OSError, ValueError, RuntimeError) as exc:
            session.close()
            st.session_state.pop("live_session", None)
            st.error(f"Could not complete and save the session report: {exc}")
            return
        if session.ended_early:
            st.session_state["analysis_notice"] = (
                "The camera feed stopped early. A partial session report was saved."
            )
        st.rerun()


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #080e1d; color: #e8edf7; }
        #MainMenu, footer, [data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
        }
        [data-testid="stSidebar"] {
            background: #101a2c;
            border-right: 1px solid #233149;
        }
        [data-testid="stHeader"] { background: rgba(8, 14, 29, 0.9); }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #f3f6fc; }
        h1 { font-size: 2.1rem !important; line-height: 1.2 !important; }
        h2 { font-size: 1.6rem !important; }
        h3 { font-size: 1.2rem !important; }
        [data-testid="stMetric"] {
            background: #101a2c;
            border: 1px solid #233149;
            border-radius: 12px;
            padding: 16px 18px;
        }
        [data-testid="stMetricValue"] { color: #46d69b; }
        [data-testid="stMetricLabel"] { color: #a8b5c9; }
        [data-testid="stRadio"] input { accent-color: #46d69b; }
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 9px;
            border: 1px solid #30415e;
            background: #17243b;
            color: #f3f6fc;
            min-height: 2.6rem;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            border-color: #46d69b;
            color: #46d69b;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #233149;
            border-radius: 10px;
        }
        hr { border-color: #233149; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _update_active_report() -> None:
    st.session_state["active_session_id"] = st.session_state[
        "selected_session_report"
    ]


def _render_new_analysis_controls() -> None:
    with st.sidebar.expander("Start a new analysis"):
        source = st.radio(
            "Session input",
            ("Live webcam", "Recorded video"),
            key="analysis_source",
            horizontal=True,
        )
        audio_upload = None
        if source == "Recorded video":
            audio_upload = st.file_uploader(
                "Optional separate audio file",
                type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
                help="Local speech-activity and noise analysis only; no transcription.",
                key="analysis_audio",
            )
            uploaded = st.file_uploader(
                "Classroom recording",
                type=["mp4", "avi", "mov", "mkv"],
                key="analysis_video",
            )
            max_frames = st.number_input(
                "Max frames (0 = full video)",
                min_value=0,
                value=0,
                step=100,
                key="analysis_max_frames",
            )
            if st.button(
                "Analyze video",
                type="primary",
                disabled=uploaded is None,
                key="analyze_video",
                **_full_width_kwargs(st.button),
            ):
                upload_dir = config.DATA_DIR / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                temp_path = upload_dir / f"{uuid4().hex}_{Path(uploaded.name).name}"
                audio_path = None
                session_id = (
                    f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
                    f"{uuid4().hex[:8]}"
                )
                try:
                    temp_path.write_bytes(uploaded.getbuffer())
                    if audio_upload is not None:
                        audio_path = upload_dir / (
                            f"{uuid4().hex}_{Path(audio_upload.name).name}"
                        )
                        audio_path.write_bytes(audio_upload.getbuffer())
                    with st.spinner(
                        "Analyzing the recording locally. Longer sessions may take a few minutes..."
                    ):
                        report = SessionProcessor(
                            max_frames=None if max_frames == 0 else int(max_frames)
                        ).process_video(
                            temp_path,
                            session_id=session_id,
                            audio_path=audio_path,
                        )
                finally:
                    _remove_staged_upload(temp_path)
                    _remove_staged_upload(audio_path)
                _load_reports.clear()
                st.session_state["active_session_id"] = report.session_id
                st.session_state["pending_session_selection"] = report.session_id
                st.session_state["analysis_notice"] = (
                    f"Analysis complete. Report saved to {report.report_path}."
                )
                st.rerun()
        else:
            audio_source = st.radio(
                "Live-session audio",
                ("None", "Microphone", "Audio file"),
                key="live_audio_source",
                horizontal=True,
                help=(
                    "Microphone audio is analyzed locally in memory and is never "
                    "transcribed or saved as a recording."
                ),
            )
            if audio_source == "Microphone":
                st.caption(
                    "The default system microphone will be used after you start "
                    "the session."
                )
            elif audio_source == "Audio file":
                audio_upload = st.file_uploader(
                    "Optional audio file",
                    type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
                    help="Local speech-activity and noise analysis only; no transcription.",
                    key="analysis_audio",
                )
            duration = st.slider(
                "Capture duration (seconds)",
                min_value=10,
                max_value=120,
                value=60,
                key="analysis_webcam_duration",
            )
            if st.button(
                "Start live session",
                type="primary",
                key="start_live_webcam",
                **_full_width_kwargs(st.button),
            ):
                audio_path = None
                if audio_source == "Audio file" and audio_upload is not None:
                    upload_dir = config.DATA_DIR / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    audio_path = upload_dir / (
                        f"{uuid4().hex}_{Path(audio_upload.name).name}"
                    )
                    audio_path.write_bytes(audio_upload.getbuffer())
                    st.session_state["staged_live_audio_path"] = audio_path
                try:
                    session_id = (
                        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
                        f"{uuid4().hex[:8]}"
                    )
                    st.session_state["live_session"] = LiveWebcamSession(
                        duration_seconds=float(duration),
                        audio_path=audio_path,
                        capture_microphone=audio_source == "Microphone",
                        session_id=session_id,
                    )
                    st.session_state["live_signal_history"] = []
                    st.rerun()
                except RuntimeError as exc:
                    _cleanup_staged_live_audio()
                    st.error(f"Could not start the webcam session: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="EduSense AI | Classroom Intelligence",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _render_styles()

    st.sidebar.markdown("## EduSense AI")
    st.sidebar.caption("Classroom intelligence")
    live_session = st.session_state.get("live_session")
    page = st.sidebar.radio(
        "Workspace",
        _PAGES,
        key="workspace_page",
        disabled=live_session is not None,
    )
    if live_session is None:
        _render_new_analysis_controls()
    else:
        st.sidebar.caption("End the active session before starting another analysis.")
    st.sidebar.divider()

    if live_session is not None:
        st.sidebar.success(
            f"SESSION ACTIVE · {_format_elapsed(live_session.elapsed_seconds)}"
        )
        st.sidebar.metric(
            "Students present",
            sum(record.present for record in live_session.current_attendance),
        )
        if st.sidebar.button(
            "End Session",
            type="primary",
            key="end_live_session_sidebar",
            **_full_width_kwargs(st.sidebar.button),
        ):
            try:
                _finish_live_session(live_session)
            except (OSError, ValueError, RuntimeError) as exc:
                live_session.close()
                _cleanup_staged_live_audio()
                st.session_state.pop("live_session", None)
                st.sidebar.error(f"Could not save the session report: {exc}")
            else:
                st.rerun()

    reports = _load_reports()
    report_ids = list(reports)
    pending_selection = st.session_state.pop("pending_session_selection", None)
    if pending_selection in reports:
        st.session_state["selected_session_report"] = pending_selection
    preferred_id = st.session_state.get("active_session_id")
    if preferred_id not in reports:
        preferred_id = report_ids[0] if report_ids else None

    if report_ids:
        selected_id = st.sidebar.selectbox(
            "Session report",
            report_ids,
            index=report_ids.index(preferred_id),
            key="selected_session_report",
            on_change=_update_active_report,
        )
        report = reports[selected_id]
        st.sidebar.caption(f"{len(report_ids)} local report(s)")
    else:
        selected_id = None
        report = None
        st.sidebar.info("Analyze a video or webcam session to create a report.")

    st.sidebar.caption(
        "Processed locally · no emotion recognition or speech transcription"
    )
    notice = st.session_state.pop("analysis_notice", None)
    if notice:
        st.success(notice)

    if live_session is not None:
        st.title("Live Classroom Dashboard")
        st.caption(
            "Face tracking and visual engagement are processed locally. "
            "Participant IDs are temporary and session-specific."
        )
        st.success(
            f"LIVE SESSION · {_format_elapsed(live_session.elapsed_seconds)} elapsed"
        )
        _render_live_session(live_session)
        return

    if report is None:
        st.title("Classroom Dashboard")
        st.caption(
            "Live attendance and engagement · locally processed, privacy-conscious"
        )
        st.info("No session selected yet. Start a live webcam session or analyze a recording.")
        status_column, features_column = st.columns(2)
        with status_column:
            st.markdown("### Session status")
            st.metric("Session status", "Not started")
            st.metric("Saved reports", len(reports))
        with features_column:
            st.markdown("### What this platform measures")
            st.markdown(
                "- Sustained attendance with session-local participant IDs\n"
                "- Head pose, movement, and inactivity signals\n"
                "- Optional speech-activity and ambient-noise proxies\n"
                "- Explainable session trends and teaching suggestions"
            )
        st.caption(
            "No emotion recognition, speech transcription, learner grades, "
            "or participant rankings."
        )
        return

    if page == "Dashboard":
        _render_dashboard(selected_id, report)
    elif page == "Teacher Central":
        _render_teacher_central(selected_id, report, reports)
    else:
        _render_analytics(selected_id, report)


if __name__ == "__main__":
    main()
