from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_ANGLE_RE = re.compile(r"^PROCESSING ANGLE:\s*(?P<angle>\d+)°\s*$")
_FOUND_PAIRS_RE = re.compile(r"^Found\s+(?P<count>\d+)\s+matching sample pairs\s*$")
_SAMPLE_RE = re.compile(r"^Processing sample\s+(?P<idx>\d+)\/(?P<total>\d+):\s+(?P<name>.+?)\s*$")
_SAMPLE_TYPE_RE = re.compile(
    r"^\s*Sample type:\s*(?P<type>[A-Z]+)\s*\(has_weapons=(?P<has_weapons>True|False)\)\s*$"
)
_SYNC_RE = re.compile(r"^\s*Synchronized:\s*(?P<pairs>\d+)\s+frame pairs\s*$")
_COMPLETED_FRAMES_RE = re.compile(r"^\s*✓\s+Completed all\s+(?P<frames>\d+)\s+frames\s*$")
_FRAME_PEOPLE_RE = re.compile(
    r"^\s*Frame\s+(?P<frame>\d+):\s*D1=(?P<d1>\d+)\s+people,\s+D2=(?P<d2>\d+)\s+people\s*$"
)
_FUSED_RE = re.compile(
    r"^\s*Frame\s+(?P<frame>\d+):\s*Fused\s+(?P<fused>\d+)\s+detections\s+from both drones,\s+total\s+(?P<total>\d+)\s+detections\s*$"
)
_SECTION_RE = re.compile(r"^---\s*(?P<section>[^-].*?)\s*---\s*$")

_OVERALL_METRICS_HDR_RE = re.compile(r"^\s*OVERALL METRICS:\s*$")
_PER_SAMPLE_METRICS_HDR_RE = re.compile(r"^\s*PER-SAMPLE METRICS\b.*:\s*$")
_PER_SAMPLE_BY_CLASS_HDR_RE = re.compile(r"^\s*PER-SAMPLE METRICS BY CLASS\b.*:\s*$")
_METRICS_BY_DISTANCE_HDR_RE = re.compile(r"^\s*METRICS BY DISTANCE:\s*$")
_METRICS_BY_HEIGHT_HDR_RE = re.compile(r"^\s*METRICS BY CAMERA HEIGHT:\s*$")
_METRICS_BY_CLASS_HDR_RE = re.compile(r"^\s*METRICS BY CLASS:\s*$")

_OVERALL_DISTANCE_RMSE_RE = re.compile(
    r"^\s*OVERALL DISTANCE ESTIMATION RMSE:\s*(?P<rmse>[-+]?(?:\d+\.\d+|\d+))m\s*$"
)
_RMSE_BY_COMBO_HDR_RE = re.compile(r"^\s*RMSE BY \(DISTANCE, HEIGHT\) COMBINATIONS:\s*$")
_RMSE_BY_COMBO_PINHOLE_HDR_RE = re.compile(r"^\s*PINHOLE RMSE BY \(DISTANCE, HEIGHT\) COMBINATIONS:\s*$")
_RMSE_COMBO_RE = re.compile(
    r"^\s*Distance:\s*(?P<distance>[-+]?(?:\d+\.\d+|\d+))m,\s*Height:\s*(?P<height>[-+]?(?:\d+\.\d+|\d+))m\s*$"
)
_RMSE_CLASS_ALL_RE = re.compile(
    r"^\s*Class 'all':\s*RMSE\s*=\s*(?P<rmse>[-+]?(?:\d+\.\d+|\d+))m\b.*$"
)
_RMSE_METHOD_HDR_RE = re.compile(r"^\s*DISTANCE ESTIMATION METHOD COMPARISON:\s*$")
_PINHOLE_RMSE_RE = re.compile(r"^\s*PINHOLE RMSE:\s*(?P<rmse>[-+]?(?:\d+\.\d+|\d+))m\b.*$")
_PITCH_RMSE_RE = re.compile(r"^\s*PITCH-BASED RMSE:\s*(?P<rmse>[-+]?(?:\d+\.\d+|\d+))m\b.*$")
_FUSED_GEO_RMSE_RE = re.compile(
    r"^\s*FUSED-GEO DISTANCE RMSE:\s*(?P<rmse>[-+]?(?:\d+\.\d+|\d+))m\b.*$"
)

_TOTAL_IMAGES_PROCESSED_RE = re.compile(r"^\s*Total images processed:\s*(?P<count>\d+)\s*$")
_TOTAL_SAMPLES_PROCESSED_RE = re.compile(r"^\s*Total samples processed:\s*(?P<count>\d+)\s*$")

# Key/value-ish lines found in the end-of-run metrics.
_METRIC_FLOAT_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9\- ]+):\s*(?P<val>[-+]?(?:\d+\.\d+|\d+))(?:\s*(?P<unit>\w+))?\s*$")
_METRIC_INTS_RE = re.compile(
    r"^\s*(?P<key>TP|TN|FP|FN):\s*(?P<val>\d+)\s*$"
)
_TP_TN_FP_FN_LINE_RE = re.compile(
    r"^\s*TP:\s*(?P<tp>\d+),\s*TN:\s*(?P<tn>\d+),\s*FP:\s*(?P<fp>\d+),\s*FN:\s*(?P<fn>\d+)\s*$"
)

def _safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


@dataclass
class RunningStats:
    values: List[float] = field(default_factory=list)

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        self.values.append(value)

    def summary(self) -> Dict[str, Any]:
        if not self.values:
            return {"count": 0}
        vals = self.values
        out: Dict[str, Any] = {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.fmean(vals),
        }
        if len(vals) >= 2:
            out["stdev"] = statistics.pstdev(vals)
        return out


@dataclass
class AngleSummary:
    angle_deg: int
    sample_pairs: Optional[int] = None
    samples_total_expected: Optional[int] = None
    samples_started: int = 0
    samples_completed: int = 0
    frame_pairs_expected_per_sample: RunningStats = field(default_factory=RunningStats)
    completed_frames_per_sample: RunningStats = field(default_factory=RunningStats)
    d1_people_per_frame: RunningStats = field(default_factory=RunningStats)
    d2_people_per_frame: RunningStats = field(default_factory=RunningStats)
    fused_detections_per_frame: RunningStats = field(default_factory=RunningStats)
    total_detections_per_frame: RunningStats = field(default_factory=RunningStats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angle_deg": self.angle_deg,
            "sample_pairs": self.sample_pairs,
            "samples_total_expected": self.samples_total_expected,
            "samples_started": self.samples_started,
            "samples_completed": self.samples_completed,
            "frame_pairs_expected_per_sample": self.frame_pairs_expected_per_sample.summary(),
            "completed_frames_per_sample": self.completed_frames_per_sample.summary(),
            "d1_people_per_frame": self.d1_people_per_frame.summary(),
            "d2_people_per_frame": self.d2_people_per_frame.summary(),
            "fused_detections_per_frame": self.fused_detections_per_frame.summary(),
            "total_detections_per_frame": self.total_detections_per_frame.summary(),
        }


@dataclass
class LogSummary:
    log_path: Path
    angles: Dict[int, AngleSummary] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # Metrics keyed by angle -> setup (Drone 1/Drone 2/Fused) -> metrics.
    metrics: Dict[int, Dict[str, "SetupMetrics"]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_file": str(self.log_path),
            "angles": {str(k): v.to_dict() for k, v in sorted(self.angles.items())},
            "notes": self.notes,
            "metrics": {
                str(angle): {setup: m.to_dict() for setup, m in setups.items()}
                for angle, setups in sorted(self.metrics.items())
            },
        }


@dataclass
class SetupMetrics:
    overall: Dict[str, Any] = field(default_factory=dict)
    per_sample: Dict[str, Any] = field(default_factory=dict)
    total_images_processed: Optional[int] = None
    total_samples_processed: Optional[int] = None
    rmse_overall_m: Optional[float] = None
    rmse_methods_m: Dict[str, float] = field(default_factory=dict)  # keys: 'pinhole', 'pitch'
    rmse_by_combo_m: Dict[str, float] = field(default_factory=dict)  # key: 'distance,height' like '5.0,2.0'
    rmse_pinhole_by_combo_m: Dict[str, float] = field(default_factory=dict)  # parsed from PINHOLE RMSE BY ...
    fused_geo_rmse_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "per_sample": self.per_sample,
            "total_images_processed": self.total_images_processed,
            "total_samples_processed": self.total_samples_processed,
            "rmse_overall_m": self.rmse_overall_m,
            "rmse_methods_m": self.rmse_methods_m,
            "rmse_by_combo_m": self.rmse_by_combo_m,
            "rmse_pinhole_by_combo_m": self.rmse_pinhole_by_combo_m,
            "fused_geo_rmse_m": self.fused_geo_rmse_m,
        }


def _get_or_create_setup_metrics(summary: LogSummary, angle_deg: int, setup: str) -> SetupMetrics:
    summary.metrics.setdefault(angle_deg, {})
    if setup not in summary.metrics[angle_deg]:
        summary.metrics[angle_deg][setup] = SetupMetrics()
    return summary.metrics[angle_deg][setup]


def _get_or_create_angle(summary: LogSummary, angle_deg: int) -> AngleSummary:
    if angle_deg not in summary.angles:
        summary.angles[angle_deg] = AngleSummary(angle_deg=angle_deg)
    return summary.angles[angle_deg]


def _parse_kv_line_into(line: str, out: Dict[str, Any]) -> bool:
    """Parse a single metrics line into out; returns True if parsed."""
    m = _TP_TN_FP_FN_LINE_RE.match(line)
    if m:
        out.update({
            "TP": int(m.group("tp")),
            "TN": int(m.group("tn")),
            "FP": int(m.group("fp")),
            "FN": int(m.group("fn")),
        })
        return True

    m = _METRIC_INTS_RE.match(line)
    if m:
        out[m.group("key")] = int(m.group("val"))
        return True

    m = _METRIC_FLOAT_RE.match(line)
    if m:
        key = re.sub(r"\s+", " ", m.group("key").strip())
        val = _safe_float(m.group("val"))
        if val is None:
            return False
        out[key] = val
        return True

    return False


def summarize_log_file(path: Path) -> LogSummary:
    summary = LogSummary(log_path=path)

    current_angle: Optional[int] = None
    current_setup: Optional[str] = None
    current_metric_scope: Optional[str] = None  # 'overall' | 'per_sample' | None
    in_rmse_combo: bool = False
    rmse_combo_target: str = "primary"  # 'primary' | 'pinhole'
    in_rmse_methods: bool = False
    current_combo_key: Optional[str] = None

    # Per-sample tracking (within an angle).
    current_sample_total_frames: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if "Windows compatibility workaround" in line:
                summary.notes.append(line.strip())

            m = _ANGLE_RE.match(line.strip())
            if m:
                current_angle = int(m.group("angle"))
                _get_or_create_angle(summary, current_angle)
                current_setup = None
                current_metric_scope = None
                in_rmse_combo = False
                rmse_combo_target = "primary"
                in_rmse_methods = False
                current_combo_key = None
                continue

            m = _FOUND_PAIRS_RE.match(line.strip())
            if m and current_angle is not None:
                _get_or_create_angle(summary, current_angle).sample_pairs = int(m.group("count"))
                continue

            m = _SAMPLE_RE.match(line.strip())
            if m and current_angle is not None:
                angle = _get_or_create_angle(summary, current_angle)
                angle.samples_started += 1
                angle.samples_total_expected = int(m.group("total"))
                current_sample_total_frames = None
                continue

            m = _SYNC_RE.match(line)
            if m and current_angle is not None:
                angle = _get_or_create_angle(summary, current_angle)
                angle.frame_pairs_expected_per_sample.add(_safe_float(m.group("pairs")))
                current_sample_total_frames = _safe_int(m.group("pairs"))
                continue

            m = _COMPLETED_FRAMES_RE.match(line)
            if m and current_angle is not None:
                angle = _get_or_create_angle(summary, current_angle)
                angle.samples_completed += 1
                angle.completed_frames_per_sample.add(_safe_float(m.group("frames")))
                continue

            m = _FRAME_PEOPLE_RE.match(line)
            if m and current_angle is not None:
                angle = _get_or_create_angle(summary, current_angle)
                angle.d1_people_per_frame.add(_safe_float(m.group("d1")))
                angle.d2_people_per_frame.add(_safe_float(m.group("d2")))
                continue

            m = _FUSED_RE.match(line)
            if m and current_angle is not None:
                angle = _get_or_create_angle(summary, current_angle)
                angle.fused_detections_per_frame.add(_safe_float(m.group("fused")))
                angle.total_detections_per_frame.add(_safe_float(m.group("total")))
                continue

            m = _SECTION_RE.match(line.strip())
            if m:
                section_name = m.group("section").strip()
                if current_angle is not None:
                    # We only care about known setups.
                    normalized = section_name
                    if normalized.lower().startswith("drone"):
                        normalized = " ".join(normalized.split())  # collapse spaces
                    elif normalized.lower() == "fused":
                        normalized = "Fused"
                    current_setup = normalized
                    _get_or_create_setup_metrics(summary, current_angle, current_setup)
                current_metric_scope = None
                in_rmse_combo = False
                rmse_combo_target = "primary"
                in_rmse_methods = False
                current_combo_key = None
                continue

            # Everything below requires we are inside an angle + setup.
            if current_angle is None or current_setup is None:
                continue

            setup_metrics = _get_or_create_setup_metrics(summary, current_angle, current_setup)

            # Parse counts from the COMPREHENSIVE DETECTION STATISTICS block.
            m = _TOTAL_IMAGES_PROCESSED_RE.match(line)
            if m:
                setup_metrics.total_images_processed = int(m.group("count"))
                continue

            m = _TOTAL_SAMPLES_PROCESSED_RE.match(line)
            if m:
                setup_metrics.total_samples_processed = int(m.group("count"))
                continue

            if _OVERALL_METRICS_HDR_RE.match(line):
                current_metric_scope = "overall"
                in_rmse_combo = False
                rmse_combo_target = "primary"
                in_rmse_methods = False
                current_combo_key = None
                continue

            if _PER_SAMPLE_METRICS_HDR_RE.match(line):
                current_metric_scope = "per_sample"
                in_rmse_combo = False
                rmse_combo_target = "primary"
                in_rmse_methods = False
                current_combo_key = None
                continue

            if _PER_SAMPLE_BY_CLASS_HDR_RE.match(line):
                current_metric_scope = None
                continue

            if _METRICS_BY_DISTANCE_HDR_RE.match(line) or _METRICS_BY_HEIGHT_HDR_RE.match(line) or _METRICS_BY_CLASS_HDR_RE.match(line):
                current_metric_scope = None
                continue

            m = _FUSED_GEO_RMSE_RE.match(line)
            if m:
                setup_metrics.fused_geo_rmse_m = float(m.group("rmse"))
                continue

            m = _OVERALL_DISTANCE_RMSE_RE.match(line)
            if m:
                setup_metrics.rmse_overall_m = float(m.group("rmse"))
                continue

            if _RMSE_BY_COMBO_HDR_RE.match(line):
                in_rmse_combo = True
                rmse_combo_target = "primary"
                in_rmse_methods = False
                current_metric_scope = None
                current_combo_key = None
                continue

            if _RMSE_BY_COMBO_PINHOLE_HDR_RE.match(line):
                in_rmse_combo = True
                rmse_combo_target = "pinhole"
                in_rmse_methods = False
                current_metric_scope = None
                current_combo_key = None
                continue

            if _RMSE_METHOD_HDR_RE.match(line):
                in_rmse_methods = True
                in_rmse_combo = False
                current_metric_scope = None
                current_combo_key = None
                continue

            if in_rmse_combo:
                m = _RMSE_COMBO_RE.match(line)
                if m:
                    dist = float(m.group("distance"))
                    height = float(m.group("height"))
                    current_combo_key = f"{dist},{height}"
                    continue

                if current_combo_key is not None:
                    m = _RMSE_CLASS_ALL_RE.match(line)
                    if m:
                        if rmse_combo_target == "pinhole":
                            setup_metrics.rmse_pinhole_by_combo_m[current_combo_key] = float(m.group("rmse"))
                        else:
                            setup_metrics.rmse_by_combo_m[current_combo_key] = float(m.group("rmse"))
                        continue

            if in_rmse_methods:
                m = _PINHOLE_RMSE_RE.match(line)
                if m:
                    setup_metrics.rmse_methods_m["pinhole"] = float(m.group("rmse"))
                    continue
                m = _PITCH_RMSE_RE.match(line)
                if m:
                    setup_metrics.rmse_methods_m["pitch"] = float(m.group("rmse"))
                    continue

            if current_metric_scope == "overall":
                _parse_kv_line_into(line, setup_metrics.overall)
                continue

            if current_metric_scope == "per_sample":
                _parse_kv_line_into(line, setup_metrics.per_sample)
                continue

    return summary


def _format_angle(angle: AngleSummary) -> str:
    pairs = angle.sample_pairs
    started = angle.samples_started
    completed = angle.samples_completed
    expected = angle.samples_total_expected

    frame_pairs = angle.frame_pairs_expected_per_sample.summary()
    frames_done = angle.completed_frames_per_sample.summary()

    d1_people = angle.d1_people_per_frame.summary()
    d2_people = angle.d2_people_per_frame.summary()

    fused_det = angle.fused_detections_per_frame.summary()

    parts = [f"Angle {angle.angle_deg}°"]
    if pairs is not None:
        parts.append(f"sample_pairs={pairs}")
    if expected is not None:
        parts.append(f"samples={started}/{expected} started, {completed}/{expected} completed")
    else:
        parts.append(f"samples_started={started}, samples_completed={completed}")

    if frame_pairs.get("count", 0):
        parts.append(f"frame_pairs/sample≈{frame_pairs.get('mean'):.1f}")
    if frames_done.get("count", 0):
        parts.append(f"frames_done/sample≈{frames_done.get('mean'):.1f}")

    if d1_people.get("count", 0) and d2_people.get("count", 0):
        parts.append(f"people/frame d1≈{d1_people.get('mean'):.2f}, d2≈{d2_people.get('mean'):.2f}")

    if fused_det.get("count", 0):
        parts.append(f"fused_dets/frame≈{fused_det.get('mean'):.2f}")

    return " | ".join(parts)


def render_text_summary(summary: LogSummary) -> str:
    lines: List[str] = []
    lines.append(f"Log: {summary.log_path.name}")

    if summary.notes:
        lines.append("Notes:")
        for note in summary.notes:
            lines.append(f"  - {note}")

    if summary.angles:
        lines.append("Angles:")
        for angle_deg in sorted(summary.angles.keys()):
            lines.append(f"  - {_format_angle(summary.angles[angle_deg])}")

    # Show a compact view of metrics if available.
    if summary.metrics:
        lines.append("Metrics:")
        for angle_deg in sorted(summary.metrics.keys()):
            for setup in ["Drone 1", "Drone 2", "Fused"]:
                if setup not in summary.metrics[angle_deg]:
                    continue
                m = summary.metrics[angle_deg][setup]
                if m.overall:
                    f1 = m.overall.get("F1-Score")
                    acc = m.overall.get("Accuracy")
                    lines.append(f"  - angle={angle_deg} setup={setup} overall: acc={acc} f1={f1}")
                if m.per_sample:
                    f1 = m.per_sample.get("F1-Score")
                    acc = m.per_sample.get("Accuracy")
                    lines.append(f"  - angle={angle_deg} setup={setup} per-sample: acc={acc} f1={f1}")

    return "\n".join(lines) + "\n"


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _fmt_metric(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, (float,)):
        return f"{value:.{decimals}f}"
    return str(value)


def render_latex_tables(summaries: List[LogSummary]) -> str:
    """Render LaTeX tables for the provided summaries.

    Produces:
    - Detection table (frame + sample levels), with angle encoded in Setup.
    - RMSE table (combo pitch-based + overall methods), grouped by angle.
    """

    # For now, merge across log files by just concatenating rows per (log, angle).
    det_rows: List[str] = []
    rmse_rows: List[str] = []

    for s in summaries:
        for angle_deg, setups in sorted(s.metrics.items()):
            for setup_name in ["Drone 1", "Drone 2", "Fused"]:
                if setup_name not in setups:
                    continue
                m = setups[setup_name]
                setup_label = _latex_escape(f"{setup_name} ({angle_deg}°)")

                def total_from(d: Dict[str, Any]) -> Optional[int]:
                    try:
                        tp = int(d.get("TP"))
                        tn = int(d.get("TN"))
                        fp = int(d.get("FP"))
                        fn = int(d.get("FN"))
                        return tp + tn + fp + fn
                    except Exception:
                        return None

                if m.overall:
                    total = m.total_images_processed if m.total_images_processed is not None else total_from(m.overall)
                    det_rows.append(
                        " ".join(
                            [
                                "Frame",
                                "&",
                                setup_label,
                                "&",
                                _fmt_metric(m.overall.get("Accuracy")),
                                "&",
                                _fmt_metric(m.overall.get("Precision")),
                                "&",
                                _fmt_metric(m.overall.get("Recall")),
                                "&",
                                _fmt_metric(m.overall.get("F1-Score")),
                                "&",
                                _fmt_metric(m.overall.get("TP"), decimals=0),
                                "&",
                                _fmt_metric(m.overall.get("FP"), decimals=0),
                                "&",
                                _fmt_metric(m.overall.get("FN"), decimals=0),
                                "&",
                                _fmt_metric(m.overall.get("TN"), decimals=0),
                                "&",
                                _fmt_metric(total, decimals=0),
                                r"\\",
                            ]
                        )
                    )

                if m.per_sample:
                    total = m.total_samples_processed if m.total_samples_processed is not None else total_from(m.per_sample)
                    det_rows.append(
                        " ".join(
                            [
                                "Sample",
                                "&",
                                setup_label,
                                "&",
                                _fmt_metric(m.per_sample.get("Accuracy")),
                                "&",
                                _fmt_metric(m.per_sample.get("Precision")),
                                "&",
                                _fmt_metric(m.per_sample.get("Recall")),
                                "&",
                                _fmt_metric(m.per_sample.get("F1-Score")),
                                "&",
                                _fmt_metric(m.per_sample.get("TP"), decimals=0),
                                "&",
                                _fmt_metric(m.per_sample.get("FP"), decimals=0),
                                "&",
                                _fmt_metric(m.per_sample.get("FN"), decimals=0),
                                "&",
                                _fmt_metric(m.per_sample.get("TN"), decimals=0),
                                "&",
                                _fmt_metric(total, decimals=0),
                                r"\\",
                            ]
                        )
                    )

            # RMSE rows: by (distance,height) combos using Drone 1 and Drone 2.
            d1 = setups.get("Drone 1")
            d2 = setups.get("Drone 2")
            if d1 and d2:
                # combo keys like '5.0,2.0'
                common_combo_keys = sorted(set(d1.rmse_by_combo_m.keys()) & set(d2.rmse_by_combo_m.keys()))
                for ck in common_combo_keys:
                    dist_s, height_s = ck.split(",")
                    rmse_rows.append(
                        " ".join(
                            [
                                "Pitch-based",
                                "&",
                                str(angle_deg),
                                "&",
                                dist_s,
                                "&",
                                height_s,
                                "&",
                                _fmt_metric(d1.rmse_by_combo_m.get(ck)),
                                "&",
                                _fmt_metric(d2.rmse_by_combo_m.get(ck)),
                                r"\\",
                            ]
                        )
                    )

                # Overall method comparison, if present.
                if "pinhole" in d1.rmse_methods_m and "pinhole" in d2.rmse_methods_m:
                    rmse_rows.append(
                        " ".join(
                            [
                                "Pinhole (overall)",
                                "&",
                                str(angle_deg),
                                "&",
                                "all",
                                "&",
                                "all",
                                "&",
                                _fmt_metric(d1.rmse_methods_m.get("pinhole")),
                                "&",
                                _fmt_metric(d2.rmse_methods_m.get("pinhole")),
                                r"\\",
                            ]
                        )
                    )
                if "pitch" in d1.rmse_methods_m and "pitch" in d2.rmse_methods_m:
                    rmse_rows.append(
                        " ".join(
                            [
                                "Pitch-based (overall)",
                                "&",
                                str(angle_deg),
                                "&",
                                "all",
                                "&",
                                "all",
                                "&",
                                _fmt_metric(d1.rmse_methods_m.get("pitch")),
                                "&",
                                _fmt_metric(d2.rmse_methods_m.get("pitch")),
                                r"\\",
                            ]
                        )
                    )

    det_table = "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Detection performance at frame and sample levels for single-UAV and fused configurations.}",
            r"\label{tab:detection_combined}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{l l c c c c c c c c c}",
            r"\hline",
            r"\textbf{Level} & \textbf{Setup} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{TN} & \textbf{Total} \\",
            r"\hline",
            r"\hline",
            *(det_rows if det_rows else [r"% (no metrics found)"]),
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )

    rmse_table = "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{RMSE (meters) by distance, altitude, viewing angle, and estimator.}",
            r"\label{tab:rmse_detailed}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{l c c c c c}",
            r"\hline",
            r"\textbf{Estimator} & \textbf{Angle} & \textbf{Distance} & \textbf{Height} & \textbf{Drone 1} & \textbf{Drone 2} \\",
            r"\hline",
            *(rmse_rows if rmse_rows else [r"% (no RMSE data found)\\"]),
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )

    return det_table + "\n\n" + rmse_table + "\n"


def iter_log_files(log_dir: Path, pattern: str) -> List[Path]:
    # pattern may include directories (glob). If it contains a path separator, treat as glob from workspace root.
    # Otherwise, glob inside log_dir.
    if any(sep in pattern for sep in ("/", "\\")):
        return sorted(Path().glob(pattern))
    return sorted(log_dir.glob(pattern))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize console log files from drone_yolo_detection_old")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory containing .log files")
    parser.add_argument("--pattern", type=str, default="run_*_console.log", help="Glob pattern within --log-dir")
    parser.add_argument("--format", choices=["text", "json", "latex"], default="text", help="Output format")
    parser.add_argument("--out", type=Path, default=None, help="Write output to this file instead of stdout")
    args = parser.parse_args(argv)

    log_dir: Path = args.log_dir
    if not log_dir.exists():
        raise SystemExit(f"log dir not found: {log_dir}")

    log_files = iter_log_files(log_dir, args.pattern)
    if not log_files:
        raise SystemExit(f"no log files matched: {log_dir / args.pattern}")

    summaries = [summarize_log_file(p) for p in log_files]

    if args.format == "json":
        payload = [s.to_dict() for s in summaries]
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    elif args.format == "latex":
        text = render_latex_tables(summaries)
    else:
        text = "\n".join(render_text_summary(s) for s in summaries)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
