"""Generate local PDF and CSV exports for session reports."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


_CSV_FIELDS = [
    "record_type",
    "session_id",
    "start_time",
    "end_time",
    "engagement_score",
    "engagement_level",
    "visual_attention",
    "head_pose_score",
    "audio_activity",
    "noise_level",
    "track_id",
    "attended",
    "visible_seconds",
    "engagement_consistency",
    "details",
]


def build_session_csv(session_id: str, payload: Dict[str, Any]) -> str:
    """Return a CSV containing interval analytics, attendance, and suggestions."""
    analysis = payload.get("analysis", {})
    rows: List[Dict[str, Any]] = []
    intervals = analysis.get("fused_intervals", [])
    timeline = analysis.get("engagement_timeline", [])

    for index, interval in enumerate(intervals):
        metadata = interval.get("metadata", {})
        point = timeline[index] if index < len(timeline) else {}
        rows.append(
            {
                "record_type": "engagement_interval",
                "session_id": session_id,
                "start_time": interval.get("start_time"),
                "end_time": interval.get("end_time"),
                "engagement_score": interval.get("combined_score"),
                "engagement_level": point.get("level", ""),
                "visual_attention": interval.get("visual_score"),
                "head_pose_score": metadata.get("head_pose_score"),
                "audio_activity": interval.get("audio_score"),
                "noise_level": metadata.get("avg_noise"),
                "details": point.get("rationale", ""),
            }
        )

    for record in analysis.get("attendance", []):
        rows.append(
            {
                "record_type": "attendance",
                "session_id": session_id,
                "track_id": record.get("track_id"),
                "attended": record.get("present"),
                "visible_seconds": record.get("total_visible_seconds"),
                "details": (
                    f"first_seen={record.get('first_seen')}; "
                    f"last_seen={record.get('last_seen')}"
                    + (
                        f"; marked_present_at={record.get('marked_present_at')}"
                        if record.get("marked_present_at") is not None
                        else ""
                    )
                    + (
                        f"; threshold_seconds={record.get('threshold_seconds')}"
                        if record.get("threshold_seconds") is not None
                        else ""
                    )
                ),
            }
        )

    for event in payload.get("attendance_log", []):
        rows.append(
            {
                "record_type": "attendance_event",
                "session_id": session_id,
                "start_time": event.get("session_time_s"),
                "track_id": event.get("track_id"),
                "details": (
                    f"{event.get('event', '')} | {event.get('session_time', '')} | "
                    f"{event.get('timestamp_utc') or 'no clock time'} | "
                    f"{event.get('details', '')}"
                ),
            }
        )

    for event in analysis.get("events", []):
        rows.append(
            {
                "record_type": "session_event",
                "session_id": session_id,
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "details": f"{event.get('event_type', '')} | {event.get('detail', '')}",
            }
        )

    for suggestion in payload.get("teaching_suggestions", []):
        rows.append(
            {
                "record_type": "teaching_suggestion",
                "session_id": session_id,
                "details": (
                    f"{suggestion.get('category', '')} | "
                    f"{suggestion.get('time_range', '')} | "
                    f"{suggestion.get('message', '')} | "
                    f"{suggestion.get('evidence', '')}"
                ),
            }
        )

    for summary in payload.get("student_analytics", []):
        rows.append(
            {
                "record_type": "participant_reflection",
                "session_id": session_id,
                "track_id": summary.get("track_id"),
                "attended": summary.get("attended"),
                "visible_seconds": summary.get("visible_seconds"),
                "engagement_consistency": summary.get("engagement_consistency"),
                "details": (
                    f"average_attention={summary.get('avg_attention')}; "
                    f"{summary.get('reflection_note', '')}"
                ),
            }
        )

    audio = payload.get("audio_summary") or {}
    if audio.get("available"):
        rows.append(
            {
                "record_type": "audio_summary",
                "session_id": session_id,
                "noise_level": audio.get("mean_noise_db"),
                "details": (
                    f"speech_ratio={audio.get('speech_ratio')}; "
                    f"background_noise_ratio={audio.get('background_noise_ratio')}; "
                    f"silence_ratio={audio.get('silence_ratio')}; "
                    f"mean_noise_db={audio.get('mean_noise_db')}; "
                    f"speech_bursts={audio.get('speech_bursts')}"
                ),
            }
        )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(value)), style)


def _styled_table(rows: List[List[Any]], widths: List[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17243b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f2f5fa")),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#142033")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ccd5e2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_session_pdf(session_id: str, payload: Dict[str, Any]) -> bytes:
    """Return a printable, privacy-conscious PDF summary of a session."""
    analysis = payload.get("analysis", {})
    attendance = analysis.get("attendance", [])
    intervals = analysis.get("fused_intervals", [])
    suggestions = payload.get("teaching_suggestions", [])
    duration = analysis.get("session_duration_seconds", payload.get("session_duration_seconds", 0))
    present_count = sum(1 for record in attendance if record.get("present"))
    source = "Video file" if payload.get("video") else payload.get("source", "Unknown")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#17243b"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        textColor=colors.HexColor("#52627a"),
        alignment=TA_CENTER,
        spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#17243b"),
        spaceBefore=14,
        spaceAfter=7,
    )
    cell_style = ParagraphStyle(
        "ReportCell",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )
    header_style = ParagraphStyle(
        "ReportHeader",
        parent=cell_style,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    note_style = ParagraphStyle(
        "PrivacyNote",
        parent=styles["BodyText"],
        textColor=colors.HexColor("#52627a"),
        fontSize=8,
        leading=11,
        spaceBefore=12,
    )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"Session report {session_id}",
        author="EduSense AI",
    )
    story: List[Any] = [
        _paragraph("EduSense AI | Session Report", title_style),
        _paragraph("Classroom engagement and attendance summary", subtitle_style),
        _paragraph("Session overview", heading_style),
        _styled_table(
            [
                [
                    _paragraph("Session", header_style),
                    _paragraph("Source", header_style),
                    _paragraph("Duration", header_style),
                    _paragraph("Participants present", header_style),
                ],
                [
                    _paragraph(session_id, cell_style),
                    _paragraph(source, cell_style),
                    _paragraph(f"{float(duration):.1f} seconds", cell_style),
                    _paragraph(f"{present_count} of {len(attendance)}", cell_style),
                ],
            ],
            [2.2 * inch, 1.4 * inch, 1.35 * inch, 1.45 * inch],
        ),
        _paragraph("Attendance", heading_style),
    ]

    attendance_rows: List[List[Any]] = [
        [
            _paragraph("Participant", header_style),
            _paragraph("Present", header_style),
            _paragraph("Visible time (s)", header_style),
            _paragraph("Marked present at (s)", header_style),
            _paragraph("First seen (s)", header_style),
            _paragraph("Last seen (s)", header_style),
        ]
    ]
    attendance_rows.extend(
        [
            _paragraph(f"Participant {record.get('track_id', '')}", cell_style),
            _paragraph("Yes" if record.get("present") else "No", cell_style),
            _paragraph(record.get("total_visible_seconds", 0), cell_style),
            _paragraph(
                record.get("marked_present_at")
                if record.get("marked_present_at") is not None
                else "-",
                cell_style,
            ),
            _paragraph(record.get("first_seen", ""), cell_style),
            _paragraph(record.get("last_seen", ""), cell_style),
        ]
        for record in attendance
    )
    if len(attendance_rows) == 1:
        attendance_rows.append([_paragraph("No attendance records", cell_style)] + [""] * 5)
    story.append(
        _styled_table(
            attendance_rows,
            [1.2 * inch, 0.7 * inch, 1.15 * inch, 1.5 * inch, 1.1 * inch, 1.1 * inch],
        )
    )
    threshold = payload.get("attendance_threshold_seconds")
    if threshold is not None:
        story.append(
            _paragraph(
                f"Presence rule: a participant counts as present after {float(threshold):g} "
                "seconds of accumulated visibility (never from a single frame).",
                note_style,
            )
        )

    story.append(_paragraph("Engagement timeline", heading_style))
    interval_rows: List[List[Any]] = [
        [
            _paragraph("Time (s)", header_style),
            _paragraph("Engagement", header_style),
            _paragraph("Visual attention", header_style),
            _paragraph("Head pose", header_style),
            _paragraph("Audio activity", header_style),
            _paragraph("Why this score", header_style),
        ]
    ]
    timeline = analysis.get("engagement_timeline", [])
    for index, interval in enumerate(intervals):
        metadata = interval.get("metadata", {})
        level = timeline[index].get("level", "") if index < len(timeline) else ""
        rationale = timeline[index].get("rationale", "") if index < len(timeline) else ""
        interval_rows.append(
            [
                _paragraph(
                    f"{float(interval.get('start_time', 0)):.1f}-"
                    f"{float(interval.get('end_time', 0)):.1f}",
                    cell_style,
                ),
                _paragraph(
                    f"{float(interval.get('combined_score', 0)):.2f} {level}",
                    cell_style,
                ),
                _paragraph(f"{float(interval.get('visual_score', 0)):.2f}", cell_style),
                _paragraph(
                    f"{float(metadata.get('head_pose_score', 0)):.2f}", cell_style
                ),
                _paragraph(
                    f"{float(interval.get('audio_score', 0)):.2f}"
                    if metadata.get("sample_count_audio", 0)
                    else "n/a",
                    cell_style,
                ),
                _paragraph(rationale or "-", cell_style),
            ]
        )
    if len(interval_rows) == 1:
        interval_rows.append([_paragraph("No engagement intervals", cell_style)] + [""] * 5)
    story.append(
        _styled_table(
            interval_rows,
            [0.8 * inch, 1.0 * inch, 0.85 * inch, 0.75 * inch, 0.8 * inch, 3.2 * inch],
        )
    )

    events = analysis.get("events", [])
    if events:
        story.append(_paragraph("Session dynamics", heading_style))
        for event in events:
            story.append(
                Paragraph(
                    f"<b>{escape(str(event.get('event_type', '')).replace('_', ' '))}</b> "
                    f"({float(event.get('start_time', 0)):.0f}s-"
                    f"{float(event.get('end_time', 0)):.0f}s) - "
                    f"{escape(str(event.get('detail', '')))}",
                    cell_style,
                )
            )
            story.append(Spacer(1, 4))

    audio = payload.get("audio_summary") or {}
    story.append(_paragraph("Audio context", heading_style))
    if audio.get("available"):
        story.append(
            _styled_table(
                [
                    [
                        _paragraph("Speech", header_style),
                        _paragraph("Background noise", header_style),
                        _paragraph("Silence", header_style),
                        _paragraph("Mean level", header_style),
                        _paragraph("Speech bursts", header_style),
                    ],
                    [
                        _paragraph(f"{float(audio.get('speech_ratio', 0)):.0%}", cell_style),
                        _paragraph(
                            f"{float(audio.get('background_noise_ratio', 0)):.0%}", cell_style
                        ),
                        _paragraph(f"{float(audio.get('silence_ratio', 0)):.0%}", cell_style),
                        _paragraph(f"{float(audio.get('mean_noise_db', 0)):.0f} dBFS", cell_style),
                        _paragraph(audio.get("speech_bursts", 0), cell_style),
                    ],
                ],
                [1.2 * inch, 1.5 * inch, 1.2 * inch, 1.5 * inch, 1.4 * inch],
            )
        )
    else:
        story.append(_paragraph("No audio input was available for this session.", cell_style))

    story.append(_paragraph("Teaching suggestions", heading_style))
    if suggestions:
        for suggestion in suggestions:
            story.append(
                Paragraph(
                    f"<b>{escape(str(suggestion.get('category', '')))}</b> "
                    f"({escape(str(suggestion.get('time_range', '')))}) - "
                    f"{escape(str(suggestion.get('message', '')))} "
                    f"<font color='#52627a'>Evidence: "
                    f"{escape(str(suggestion.get('evidence', '')))}</font>",
                    cell_style,
                )
            )
            story.append(Spacer(1, 5))
    else:
        story.append(_paragraph("No suggestions were generated.", cell_style))

    reflections = payload.get("student_analytics", [])
    if reflections:
        story.append(_paragraph("Participant reflections", heading_style))
        for summary in reflections:
            story.append(
                Paragraph(
                    f"<b>Participant {escape(str(summary.get('track_id', '')))}</b> "
                    f"({escape(str(summary.get('engagement_consistency', '')))}) - "
                    f"{escape(str(summary.get('focus_highlights', '')))} "
                    f"{escape(str(summary.get('reflection_note', '')))}",
                    cell_style,
                )
            )
            story.append(Spacer(1, 4))

    story.append(
        _paragraph(
            "Privacy note: This report contains behavioral signal summaries only. "
            "It does not include emotion recognition, speech transcription, grades, "
            "or participant rankings. Track IDs are session-local and are not names.",
            note_style,
        )
    )
    document.build(story)
    return buffer.getvalue()
