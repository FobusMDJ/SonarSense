"""PDF and PNG report export for one log -- GET /logs/{id}/report.pdf and
GET /logs/{id}/map.png. Kept in its own module since these are heavier,
optional deps (reportlab, matplotlib) not needed by the rest of the API.
"""

from __future__ import annotations

import io

from src.backend import class_taxonomy


def build_map_png(log: dict, detections: list[dict]) -> bytes:
    """Simple scatter of located detections + track line, colored by
    display classification. No basemap tiles (keeps this dependency-free /
    offline-safe) -- axes are plain lat/lon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    located = [d for d in detections if d.get("geo_method") == "nav_fix"
               and d.get("lat") is not None and d.get("lon") is not None]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    if located:
        track = sorted(located, key=lambda d: d["frame_index"])
        ax.plot([d["lon"] for d in track], [d["lat"] for d in track],
                color="#888", linewidth=1, alpha=0.6, zorder=1, label="Track (approx.)")
        colors = {"Shipwreck": "#e6635b", "Pipe": "#4c80df", "Ghost Net": "#5daa72",
                  "Cylinder": "#ecac48", "Other Debris": "#8862c6", "Human": "#d62728"}
        for cls in set(class_taxonomy.display_classification(d["class_name"]) for d in located):
            pts = [d for d in located if class_taxonomy.display_classification(d["class_name"]) == cls]
            ax.scatter([d["lon"] for d in pts], [d["lat"] for d in pts],
                       label=cls, s=40, zorder=2, color=colors.get(cls, "#333"))
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No geolocated detections", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Detection map -- {log['filename']}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def build_report_pdf(log: dict, detections: list[dict]) -> bytes:
    """One-page-plus summary report: log metadata, per-class counts, and a
    detection table (id, class, confidence, priority, lat/lon)."""
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    story = [
        Paragraph("SonarSense Detection Report", styles["Title"]),
        Paragraph(f"Log: {log['filename']} ({log['id']})", styles["Normal"]),
        Paragraph(f"Status: {log['status']} | Uploaded: {log['uploaded_at']}", styles["Normal"]),
        Paragraph(f"Frames: {log.get('n_frames', 0)} | Detections: {log.get('n_detections', 0)}", styles["Normal"]),
        Spacer(1, 0.25 * inch),
    ]

    non_human = [d for d in detections if not class_taxonomy.is_human_class(d["class_name"])]
    human = [d for d in detections if class_taxonomy.is_human_class(d["class_name"])]
    if human:
        story.append(Paragraph(
            f"<b>{len(human)} human detection(s) recorded -- see safety review, excluded from the debris table below.</b>",
            styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))

    counts: dict[str, int] = {}
    for d in non_human:
        cls = class_taxonomy.display_classification(d["class_name"])
        counts[cls] = counts.get(cls, 0) + 1
    story.append(Paragraph("Counts by class: " + ", ".join(f"{k}: {v}" for k, v in counts.items()), styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    table_data = [["ID", "Class", "Confidence", "Priority", "Lat", "Lon"]]
    for d in non_human[:200]:  # cap rows for a sane PDF size
        cls = class_taxonomy.display_classification(d["class_name"])
        priority = class_taxonomy.priority_from_confidence(d["confidence_score"])
        lat = f"{d['lat']:.5f}" if d.get("lat") is not None else "-"
        lon = f"{d['lon']:.5f}" if d.get("lon") is not None else "-"
        table_data.append([d["id"][:8], cls, f"{d['confidence_score']:.1f}", priority, lat, lon])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(table)
    if len(non_human) > 200:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph(f"... and {len(non_human) - 200} more (truncated for report size).", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()
