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
_FUSED_GEO_RMSE_D1_RE = re.compile(
    r"^\s*FUSED-GEO DISTANCE RMSE \(D1\):\s*(?P<rmse>[-+]?(?:\d+\.\d+|\d+))m\b.*$"
)
_FUSED_GEO_RMSE_D2_RE = re.compile(
    r"^\s*FUSED-GEO DISTANCE RMSE \(D2\):\s*(?P<rmse>[-+]?(?:\d+\.\d+|\d+))m\b.*$"
)

_FUSION_GAIN_HDR_RE = re.compile(r"^\s*FUSION LOCALIZATION GAIN:\s*$")
_FUSION_GAIN_BY_COMBO_HDR_RE = re.compile(r"^\s*FUSION GAIN BY \(DISTANCE, HEIGHT\):\s*$")
_FUSION_GAIN_VS_RE = re.compile(
    r"^\s*VS\s+(?P<base>D1|D2):\s*base_rmse=(?P<base_rmse>[-+]?(?:\d+\.\d+|\d+))m,\s*"
    r"fused_rmse=(?P<fused_rmse>[-+]?(?:\d+\.\d+|\d+))m,\s*gain=(?P<gain>[-+]?(?:\d+\.\d+|\d+))m,\s*"
    r"gain_pct=(?P<pct>[-+]?(?:\d+\.\d+|\d+))%\s*$"
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
    fused_geo_rmse_d1_m: Optional[float] = None
    fused_geo_rmse_d2_m: Optional[float] = None

    # Fusion gain (stored under the Fused setup)
    fusion_gain_overall: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {'D1': {...}, 'D2': {...}}
    fusion_gain_by_combo: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)  # {'dist,height': {'D1': {...}, 'D2': {...}}}

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
            "fused_geo_rmse_d1_m": self.fused_geo_rmse_d1_m,
            "fused_geo_rmse_d2_m": self.fused_geo_rmse_d2_m,
            "fusion_gain_overall": self.fusion_gain_overall,
            "fusion_gain_by_combo": self.fusion_gain_by_combo,
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

    in_fusion_gain: bool = False
    in_fusion_gain_by_combo: bool = False
    fusion_combo_key: Optional[str] = None

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
                in_fusion_gain = False
                in_fusion_gain_by_combo = False
                fusion_combo_key = None
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
                in_fusion_gain = False
                in_fusion_gain_by_combo = False
                fusion_combo_key = None
                continue

            if _PER_SAMPLE_METRICS_HDR_RE.match(line):
                current_metric_scope = "per_sample"
                in_rmse_combo = False
                rmse_combo_target = "primary"
                in_rmse_methods = False
                current_combo_key = None
                in_fusion_gain = False
                in_fusion_gain_by_combo = False
                fusion_combo_key = None
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

            m = _FUSED_GEO_RMSE_D1_RE.match(line)
            if m:
                setup_metrics.fused_geo_rmse_d1_m = float(m.group("rmse"))
                continue

            m = _FUSED_GEO_RMSE_D2_RE.match(line)
            if m:
                setup_metrics.fused_geo_rmse_d2_m = float(m.group("rmse"))
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
                in_fusion_gain = False
                in_fusion_gain_by_combo = False
                fusion_combo_key = None
                continue

            if _FUSION_GAIN_HDR_RE.match(line):
                in_fusion_gain = True
                in_fusion_gain_by_combo = False
                fusion_combo_key = None
                continue

            if _FUSION_GAIN_BY_COMBO_HDR_RE.match(line):
                in_fusion_gain = True
                in_fusion_gain_by_combo = True
                fusion_combo_key = None
                continue

            if in_fusion_gain and in_fusion_gain_by_combo:
                m = _RMSE_COMBO_RE.match(line)
                if m:
                    dist = float(m.group("distance"))
                    height = float(m.group("height"))
                    fusion_combo_key = f"{dist},{height}"
                    setup_metrics.fusion_gain_by_combo.setdefault(fusion_combo_key, {})
                    continue

            if in_fusion_gain:
                m = _FUSION_GAIN_VS_RE.match(line)
                if m:
                    base = m.group("base")
                    payload = {
                        "base_rmse_m": float(m.group("base_rmse")),
                        "fused_rmse_m": float(m.group("fused_rmse")),
                        "gain_m": float(m.group("gain")),
                        "gain_pct": float(m.group("pct")),
                    }
                    if in_fusion_gain_by_combo and fusion_combo_key is not None:
                        setup_metrics.fusion_gain_by_combo.setdefault(fusion_combo_key, {})
                        setup_metrics.fusion_gain_by_combo[fusion_combo_key][base] = payload
                    else:
                        setup_metrics.fusion_gain_overall[base] = payload
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


def _bold_row(cells: List[str]) -> str:
    return " & ".join(f"\\textbf{{{c}}}" for c in cells) + r" \\"


def _row(cells: List[str], bold: bool = False) -> str:
    if bold:
        return _bold_row(cells)
    return " & ".join(cells) + r" \\"


def render_latex_tables(summaries: List[LogSummary]) -> str:
    """Render LaTeX tables matching the analyze_log.py output format.

    Produces the same 12 table types as analyze_log.render_latex:
    1.  Detection performance (frame-level)
    2.  Detection performance (sample-level)
    3.  Fused detection summary
    4.  Overall RMSE (pinhole vs pitch)
    5.  RMSE by (distance, height) — pitch-based
    6.  RMSE by (distance, height) — pinhole
    7.  RMSE by (class, distance, height) — pitch-based  (skipped if data unavailable)
    8.  Fused-geo RMSE
    9.  Fused-geo RMSE by (distance, height)  (from fusion gain combos)
    10. Fusion localization gain
    11. Confusion matrices (frame & sample)
    12. Detection by (distance, height) per setup  (skipped if data unavailable)
    """
    tables: List[str] = []

    # Collect all (angle_deg, setups_dict) across summaries
    all_angle_setups: List[Tuple[int, Dict[str, SetupMetrics]]] = []
    for s in summaries:
        for angle_deg, setups in sorted(s.metrics.items()):
            all_angle_setups.append((angle_deg, setups))

    # ── Helper: get metrics dict for a level ──
    def _level_dict(m: SetupMetrics, level: str) -> Dict[str, Any]:
        return m.overall if level == "frame" else m.per_sample

    # ── Helper: build a detection metrics table ──
    def _det_table(level: str, caption: str, label: str):
        rows = []
        for angle_deg, setups in all_angle_setups:
            for sname in ["Drone 1", "Drone 2", "Fused"]:
                if sname not in setups:
                    continue
                d = _level_dict(setups[sname], level)
                if not d:
                    continue
                setup_label = _latex_escape(f"{sname} ({angle_deg}°)")
                is_fused = sname == "Fused"
                cells = [
                    setup_label,
                    _fmt_metric(d.get("Accuracy")),
                    _fmt_metric(d.get("Precision")),
                    _fmt_metric(d.get("Recall")),
                    _fmt_metric(d.get("F1-Score")),
                ]
                rows.append(_row(cells, bold=is_fused))
            rows.append(r"\hline")
        return "\n".join([
            r"\begin{table}[ht]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{l c c c c}",
            r"\hline",
            r"\textbf{Setup} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\",
            r"\hline",
            *rows,
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ])

    # ── Helper: confusion matrix table ──
    def _cm_table(level: str, caption: str, label: str):
        rows = []
        for angle_deg, setups in all_angle_setups:
            for sname in ["Drone 1", "Drone 2", "Fused"]:
                if sname not in setups:
                    continue
                m = setups[sname]
                d = _level_dict(m, level)
                if not d:
                    continue
                setup_label = _latex_escape(f"{sname} ({angle_deg}°)")
                is_fused = sname == "Fused"
                tp = d.get("TP", 0)
                fp = d.get("FP", 0)
                fn = d.get("FN", 0)
                tn = d.get("TN", 0)
                if level == "frame":
                    total = m.total_images_processed if m.total_images_processed is not None else (tp + tn + fp + fn)
                else:
                    total = m.total_samples_processed if m.total_samples_processed is not None else (tp + tn + fp + fn)
                cells = [
                    setup_label,
                    _fmt_metric(tp, 0), _fmt_metric(fp, 0),
                    _fmt_metric(fn, 0), _fmt_metric(tn, 0),
                    _fmt_metric(total, 0),
                ]
                rows.append(_row(cells, bold=is_fused))
            rows.append(r"\hline")
        return "\n".join([
            r"\begin{table}[ht]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{l c c c c c}",
            r"\hline",
            r"\textbf{Setup} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{TN} & \textbf{Total} \\",
            r"\hline",
            r"\hline",
            *rows,
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ])

    # 1. Detection performance (frame-level)
    tables.append(_det_table(
        "frame",
        "Detection performance (frame-level).",
        "tab:detection_frame_metrics",
    ))

    # 2. Detection performance (sample-level)
    tables.append(_det_table(
        "sample",
        "Detection performance (sample-level).",
        "tab:detection_sample_metrics",
    ))

    # 3. Fused detection summary
    fused_rows = []
    for angle_deg, setups in all_angle_setups:
        if "Fused" not in setups:
            continue
        m = setups["Fused"]
        if m.overall:
            fused_rows.append(_row([
                "Frame", str(angle_deg),
                _fmt_metric(m.overall.get("Accuracy")),
                _fmt_metric(m.overall.get("Precision")),
                _fmt_metric(m.overall.get("Recall")),
                _fmt_metric(m.overall.get("F1-Score")),
            ]))
        if m.per_sample:
            fused_rows.append(_row([
                "Sample", str(angle_deg),
                _fmt_metric(m.per_sample.get("Accuracy")),
                _fmt_metric(m.per_sample.get("Precision")),
                _fmt_metric(m.per_sample.get("Recall")),
                _fmt_metric(m.per_sample.get("F1-Score")),
            ]))
    tables.append("\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Fused detection performance summary across viewing angles.}",
        r"\label{tab:detection_fused_summary}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l c c c c c}",
        r"\hline",
        r"\textbf{Level} & \textbf{Angle (°)} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\",
        r"\hline",
        *fused_rows,
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]))

    # 4. Overall RMSE table
    rmse_rows = []
    for angle_deg, setups in all_angle_setups:
        d1 = setups.get("Drone 1")
        d2 = setups.get("Drone 2")
        if not d1 or not d2:
            continue
        for est_label, key in [("Pinhole (overall)", "pinhole"), ("Pitch-based (overall)", "pitch")]:
            r_d1 = d1.rmse_methods_m.get(key)
            r_d2 = d2.rmse_methods_m.get(key)
            is_pitch = "Pitch" in est_label
            cells = [_latex_escape(est_label), str(angle_deg), _fmt_metric(r_d1), _fmt_metric(r_d2)]
            rmse_rows.append(_row(cells, bold=is_pitch))
        rmse_rows.append(r"\hline")
    tables.append("\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Overall RMSE (meters) comparison between pinhole and pitch-based estimators. Lower is better.}",
        r"\label{tab:rmse_overall}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l c c c}",
        r"\hline",
        r"\textbf{Estimator} & \textbf{Angle (°)} & \textbf{Drone 1} & \textbf{Drone 2} \\",
        r"\hline",
        *rmse_rows,
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]))

    # 5. RMSE by (distance, height) — pitch-based
    rmse_dh_rows = []
    for angle_deg, setups in all_angle_setups:
        d1 = setups.get("Drone 1")
        d2 = setups.get("Drone 2")
        if not d1 or not d2:
            continue
        all_dh = sorted(set(d1.rmse_by_combo_m.keys()) | set(d2.rmse_by_combo_m.keys()))
        for ck in all_dh:
            dist_s, height_s = ck.split(",")
            cells = [str(angle_deg), dist_s, height_s, _fmt_metric(d1.rmse_by_combo_m.get(ck)), _fmt_metric(d2.rmse_by_combo_m.get(ck))]
            rmse_dh_rows.append(_row(cells))
        rmse_dh_rows.append(r"\hline")

    if rmse_dh_rows:
        tables.append("\n".join([
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Pitch-based RMSE (meters) by distance, altitude, and viewing angle.}",
            r"\label{tab:rmse_pitch_by_combo}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{c c c c c}",
            r"\hline",
            r"\textbf{Angle (°)} & \textbf{Distance (m)} & \textbf{Height (m)} & \textbf{Drone 1} & \textbf{Drone 2} \\",
            r"\hline",
            *rmse_dh_rows,
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]))

    # 6. RMSE by (distance, height) — pinhole
    rmse_pin_rows = []
    for angle_deg, setups in all_angle_setups:
        d1 = setups.get("Drone 1")
        d2 = setups.get("Drone 2")
        if not d1 or not d2:
            continue
        all_dh = sorted(set(d1.rmse_pinhole_by_combo_m.keys()) | set(d2.rmse_pinhole_by_combo_m.keys()))
        for ck in all_dh:
            dist_s, height_s = ck.split(",")
            cells = [str(angle_deg), dist_s, height_s, _fmt_metric(d1.rmse_pinhole_by_combo_m.get(ck)), _fmt_metric(d2.rmse_pinhole_by_combo_m.get(ck))]
            rmse_pin_rows.append(_row(cells))
        rmse_pin_rows.append(r"\hline")

    if rmse_pin_rows:
        tables.append("\n".join([
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Pinhole RMSE (meters) by distance, altitude, and viewing angle.}",
            r"\label{tab:rmse_pinhole_by_combo}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{c c c c c}",
            r"\hline",
            r"\textbf{Angle (°)} & \textbf{Distance (m)} & \textbf{Height (m)} & \textbf{Drone 1} & \textbf{Drone 2} \\",
            r"\hline",
            *rmse_pin_rows,
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]))

    # 7. RMSE by (class, distance, height) — pitch-based (skipped: data not available in summarize_logs)

    # 8. Fused-geo RMSE table
    fused_rmse_rows = []
    for angle_deg, setups in all_angle_setups:
        fused = setups.get("Fused")
        if not fused:
            continue
        rd1 = fused.fused_geo_rmse_d1_m
        rd2 = fused.fused_geo_rmse_d2_m
        if rd1 is not None or rd2 is not None:
            cells = [str(angle_deg), _fmt_metric(rd1), _fmt_metric(rd2)]
            fused_rmse_rows.append(_row(cells))

    if fused_rmse_rows:
        tables.append("\n".join([
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Fused-geo RMSE (meters): distance from each drone to the triangulated geoposition.}",
            r"\label{tab:rmse_fused_geo}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{c c c}",
            r"\hline",
            r"\textbf{Angle (°)} & \textbf{D1 $\rightarrow$ Fused} & \textbf{D2 $\rightarrow$ Fused} \\",
            r"\hline",
            *fused_rmse_rows,
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]))

    # 9. Fused-geo RMSE by (distance, height) — from fusion_gain_by_combo data
    fused_dh_rows = []
    for angle_deg, setups in all_angle_setups:
        fused = setups.get("Fused")
        if not fused or not fused.fusion_gain_by_combo:
            continue
        for ck in sorted(fused.fusion_gain_by_combo.keys()):
            dist_s, height_s = ck.split(",")
            combo = fused.fusion_gain_by_combo[ck]
            rd1 = combo.get("D1", {}).get("fused_rmse_m")
            rd2 = combo.get("D2", {}).get("fused_rmse_m")
            cells = [str(angle_deg), dist_s, height_s, _fmt_metric(rd1), _fmt_metric(rd2)]
            fused_dh_rows.append(_row(cells))
        fused_dh_rows.append(r"\hline")

    if fused_dh_rows:
        tables.append("\n".join([
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Fused-geo RMSE (meters) by distance, altitude, and viewing angle.}",
            r"\label{tab:rmse_fused_geo_by_combo}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{c c c c c}",
            r"\hline",
            r"\textbf{Angle (°)} & \textbf{Distance (m)} & \textbf{Height (m)} & \textbf{D1 $\rightarrow$ Fused} & \textbf{D2 $\rightarrow$ Fused} \\",
            r"\hline",
            *fused_dh_rows,
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]))

    # 10. Fusion localization gain table
    gain_rows = []
    for angle_deg, setups in all_angle_setups:
        fused = setups.get("Fused")
        if not fused or not fused.fusion_gain_overall:
            continue
        for drone_lbl in ["D1", "D2"]:
            g = fused.fusion_gain_overall.get(drone_lbl)
            if g is None:
                continue
            cells = [
                str(angle_deg), drone_lbl,
                _fmt_metric(g.get("base_rmse_m")),
                _fmt_metric(g.get("fused_rmse_m")),
                _fmt_metric(g.get("gain_m")),
                f"{g.get('gain_pct', 0):.1f}\\%",
            ]
            gain_rows.append(_row(cells))
        gain_rows.append(r"\hline")

    if gain_rows:
        tables.append("\n".join([
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Fusion localization gain: RMSE improvement of fused geoposition over single-drone pitch-based estimation.}",
            r"\label{tab:fusion_gain}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{c c c c c c}",
            r"\hline",
            r"\textbf{Angle (°)} & \textbf{Baseline} & \textbf{Base RMSE (m)} & \textbf{Fused RMSE (m)} & \textbf{Gain (m)} & \textbf{Gain (\%)} \\",
            r"\hline",
            *gain_rows,
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]))

    # 11. Confusion matrices (frame & sample)
    tables.append(_cm_table(
        "frame",
        "Confusion matrix counts (frame-level).",
        "tab:detection_frame_cm",
    ))
    tables.append(_cm_table(
        "sample",
        "Confusion matrix counts (sample-level).",
        "tab:detection_sample_cm",
    ))

    # 12. Detection by (distance, height) per setup (skipped: per-combo detection data not available in summarize_logs)

    return "\n\n".join(tables) + "\n"


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
