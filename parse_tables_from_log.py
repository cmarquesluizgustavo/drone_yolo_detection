#!/usr/bin/env python3
"""parse_tables_from_log_v2.py

A more robust parser for your console logs.

Key fixes vs the original:
- Frame records are created when we first see a frame index (e.g. "Frame 0:") so
  DISTANCE/FUSED_* lines that come *before* the "best estimator" line are captured.
- Winner line supports BOTH formats:
    1) "Frame k | best estimator: {pinhole/pitch/} | best fusion: {avg/}"
    2) "Frame k | best estimator: pinhole (pinhole=...) | best fusion: average (avg=...)"
- Angle header does NOT require the degree symbol.

It keeps the rest of the aggregation/table logic the same as your original script.

Usage:
  python parse_tables_from_log_v2.py --log run_*.log --outdir results
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# --------------------------
# CONFIG
# --------------------------

# Vote mode:
# - "fractional": ties split vote equally, totals == number_of_frames for each config
# - "integer": each tied winner gets +1, totals can exceed number_of_frames
VOTE_MODE = "fractional"  # or "integer"

BOLD_RULE_SINGLE = "group_min"  # or "row_min"
BOLD_RULE_FUSION = "row_min"

# --------------------------
# Regexes
# --------------------------

SAMPLE_HEADER_RE = re.compile(r"Processing sample\s+\d+/\d+:\s+(.+)$")

# In your log: "Frame 0: D1=..., D2=..."
FRAME_START_RE = re.compile(r"^\s*Frame\s+(\d+):")

# Winner line format #1 (braced lists)
FRAME_WIN_BRACED_RE = re.compile(
    r"Frame\s+(\d+)\s+\|\s+best estimator:\s*\{([^}]*)\}\s+\|\s+best fusion:\s*\{([^}]*)\}",
    re.IGNORECASE,
)

# Winner line format #2 (plain names)
# Example:
#   Frame 0 | best estimator: pinhole (pinhole=0.38 pitch=0.42) | best fusion: average (avg=0.37 bi=1.23 weighted_avg=0.39)
FRAME_WIN_PLAIN_RE = re.compile(
    r"Frame\s+(\d+)\s+\|\s+best estimator:\s*([A-Za-z0-9_\-/ ]+?)\s*(?:\(|\|)\s*.*?\|\s+best fusion:\s*([A-Za-z0-9_\-/ ]+?)\s*(?:\(|$)",
    re.IGNORECASE,
)

DISTANCE_LINE_RE = re.compile(r"^\s*DISTANCE:\s*(.+)$", re.IGNORECASE)
KV_RE = re.compile(r"([A-Za-z0-9_\-]+)\s*=\s*([-+]?\d+(?:\.\d+)?)")

# Example in log:
#   FUSED_AVERAGE: dist1_from_fused_avg=..., dist2_from_fused_avg=...
FUSED_LINE_RE = re.compile(r"^\s*(FUSED_[A-Z_]+):\s*(.+)$")

ANGLE_LINE_RE = re.compile(r"PROCESSING ANGLE:\s*(\d+)")


# --------------------------
# Data structures
# --------------------------

@dataclass(frozen=True)
class ConfigKey:
    angle: int
    dist_gt: int
    height_gt: int


@dataclass
class FrameRecord:
    dist1_height: Optional[float] = None
    dist2_height: Optional[float] = None
    dist1_pitch: Optional[float] = None
    dist2_pitch: Optional[float] = None

    dist1_fused_avg: Optional[float] = None
    dist2_fused_avg: Optional[float] = None
    dist1_fused_bi: Optional[float] = None
    dist2_fused_bi: Optional[float] = None
    dist1_fused_wavg: Optional[float] = None
    dist2_fused_wavg: Optional[float] = None

    best_estimators: List[str] = None
    best_fusions: List[str] = None

    def __post_init__(self):
        if self.best_estimators is None:
            self.best_estimators = []
        if self.best_fusions is None:
            self.best_fusions = []


@dataclass
class SampleData:
    name: str
    angle: int
    dist_gt: int
    height_gt: int
    frames: List[FrameRecord]


# --------------------------
# Helpers
# --------------------------

def parse_dist_height_from_name(sample_name: str) -> Tuple[int, int]:
    parts = sample_name.strip().split("_")
    if len(parts) < 3:
        raise ValueError(f"Sample name '{sample_name}' does not look like class_dist_height_*")
    return int(parts[1]), int(parts[2])


def split_braced_list(s: str) -> List[str]:
    items = [x.strip() for x in s.split("/") if x.strip()]
    return [re.sub(r"\s+", " ", it) for it in items]


def add_votes(counter: Dict[str, float], winners: List[str], mode: str):
    if not winners:
        return
    if mode == "integer":
        for w in winners:
            counter[w] += 1.0
    elif mode == "fractional":
        frac = 1.0 / len(winners)
        for w in winners:
            counter[w] += frac
    else:
        raise ValueError("mode must be 'integer' or 'fractional'")


def rmse(values: List[float], gt: float) -> float:
    if not values:
        return float("nan")
    return math.sqrt(sum((v - gt) ** 2 for v in values) / len(values))


def mean2(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def fmt_rmse(v: float, bold: bool = False, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    s = f"{v:.{decimals}f}"
    return f"\\textbf{{{s}}}" if bold else s


def topk(counter: Dict[str, float], k: int = 3):
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:k]


# --------------------------
# Parsing
# --------------------------

def parse_log(path: str) -> List[SampleData]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    samples: List[SampleData] = []

    current_sample_name: Optional[str] = None
    current_angle: Optional[int] = None
    current_dist: Optional[int] = None
    current_height: Optional[int] = None

    # frame_index -> FrameRecord
    frames_by_idx: Dict[int, FrameRecord] = {}
    current_frame_idx: Optional[int] = None

    def flush_sample():
        nonlocal current_sample_name, current_angle, current_dist, current_height
        nonlocal frames_by_idx, current_frame_idx

        if current_sample_name is None:
            return

        # emit frames in index order
        ordered = [frames_by_idx[i] for i in sorted(frames_by_idx.keys())]
        samples.append(
            SampleData(
                name=current_sample_name,
                angle=int(current_angle),
                dist_gt=int(current_dist),
                height_gt=int(current_height),
                frames=ordered,
            )
        )

        current_sample_name = None
        current_dist = None
        current_height = None
        frames_by_idx = {}
        current_frame_idx = None

    for line_num, line in enumerate(lines, 1):
        # angle header
        m_ang = ANGLE_LINE_RE.search(line)
        if m_ang:
            flush_sample()
            current_angle = int(m_ang.group(1))
            continue

        # sample header
        sh = SAMPLE_HEADER_RE.search(line)
        if sh:
            flush_sample()
            current_sample_name = sh.group(1).strip()
            try:
                current_dist, current_height = parse_dist_height_from_name(current_sample_name)
            except Exception as e:
                print(f"[ERROR] dist/height parse failed at line {line_num}: {current_sample_name} -> {e}")
                current_sample_name = None
                continue
            if current_angle is None:
                print(
                    f"[ERROR] Angle not set before sample at line {line_num}: '{current_sample_name}'. "
                    f"Expected a 'PROCESSING ANGLE: XX' header before samples."
                )
                current_sample_name = None
                continue
            frames_by_idx = {}
            current_frame_idx = None
            continue

        if current_sample_name is None:
            continue

        # frame start (creates frame record early so we can capture DISTANCE/FUSED lines)
        m_fs = FRAME_START_RE.search(line)
        if m_fs:
            current_frame_idx = int(m_fs.group(1))
            frames_by_idx.setdefault(current_frame_idx, FrameRecord())
            continue

        # distance line
        m_dist = DISTANCE_LINE_RE.search(line)
        if m_dist and current_frame_idx is not None:
            fr = frames_by_idx.setdefault(current_frame_idx, FrameRecord())
            payload = m_dist.group(1)
            kvs = {m.group(1): float(m.group(2)) for m in KV_RE.finditer(payload)}

            fr.dist1_height = kvs.get("dist1_height-based", fr.dist1_height)
            fr.dist2_height = kvs.get("dist2_height-based", fr.dist2_height)
            fr.dist1_pitch = kvs.get("dist1_pitch-based", fr.dist1_pitch)
            fr.dist2_pitch = kvs.get("dist2_pitch-based", fr.dist2_pitch)
            continue

        # fused lines
        m_fused = FUSED_LINE_RE.search(line)
        if m_fused and current_frame_idx is not None:
            fr = frames_by_idx.setdefault(current_frame_idx, FrameRecord())
            payload = m_fused.group(2)
            kvs = {m.group(1): float(m.group(2)) for m in KV_RE.finditer(payload)}

            fr.dist1_fused_avg = kvs.get("dist1_from_fused_avg", fr.dist1_fused_avg)
            fr.dist2_fused_avg = kvs.get("dist2_from_fused_avg", fr.dist2_fused_avg)
            fr.dist1_fused_bi = kvs.get("dist1_from_fused_bi", fr.dist1_fused_bi)
            fr.dist2_fused_bi = kvs.get("dist2_from_fused_bi", fr.dist2_fused_bi)
            fr.dist1_fused_wavg = kvs.get("dist1_from_weighted_fused_avg", fr.dist1_fused_wavg)
            fr.dist2_fused_wavg = kvs.get("dist2_from_weighted_fused_avg", fr.dist2_fused_wavg)
            continue

        # winner lines (either format)
        m_win = FRAME_WIN_BRACED_RE.search(line)
        if m_win:
            idx = int(m_win.group(1))
            fr = frames_by_idx.setdefault(idx, FrameRecord())
            fr.best_estimators = split_braced_list(m_win.group(2))
            fr.best_fusions = split_braced_list(m_win.group(3))
            continue

        m_win2 = FRAME_WIN_PLAIN_RE.search(line)
        if m_win2:
            idx = int(m_win2.group(1))
            fr = frames_by_idx.setdefault(idx, FrameRecord())
            fr.best_estimators = [m_win2.group(2).strip()]
            fr.best_fusions = [m_win2.group(3).strip()]
            continue

    flush_sample()
    return samples


# --------------------------
# Aggregation
# --------------------------

class ConfigAgg:
    def __init__(self):
        self.g1: List[float] = []
        self.g2: List[float] = []
        self.h1: List[float] = []
        self.h2: List[float] = []

        # Fused localization (per-UAV and combined)
        self.bi_1: List[float] = []
        self.bi_2: List[float] = []
        self.avg_1: List[float] = []
        self.avg_2: List[float] = []
        self.wavg_1: List[float] = []
        self.wavg_2: List[float] = []

        # Mean-between-UAVs per frame (kept for existing fused table)
        self.bi: List[float] = []
        self.avg: List[float] = []
        self.wavg: List[float] = []

        self.est_votes: Dict[str, float] = defaultdict(float)
        self.fus_votes: Dict[str, float] = defaultdict(float)
        self.frames_with_votes: int = 0


def aggregate(samples: List[SampleData]) -> Dict[ConfigKey, ConfigAgg]:
    agg: Dict[ConfigKey, ConfigAgg] = defaultdict(ConfigAgg)

    for s in samples:
        key = ConfigKey(s.angle, s.dist_gt, s.height_gt)
        a = agg[key]

        for fr in s.frames:
            if fr.dist1_pitch is not None:
                a.g1.append(fr.dist1_pitch)
            if fr.dist2_pitch is not None:
                a.g2.append(fr.dist2_pitch)
            if fr.dist1_height is not None:
                a.h1.append(fr.dist1_height)
            if fr.dist2_height is not None:
                a.h2.append(fr.dist2_height)

            # Per-UAV fused distances
            if fr.dist1_fused_bi is not None:
                a.bi_1.append(fr.dist1_fused_bi)
            if fr.dist2_fused_bi is not None:
                a.bi_2.append(fr.dist2_fused_bi)
            if fr.dist1_fused_avg is not None:
                a.avg_1.append(fr.dist1_fused_avg)
            if fr.dist2_fused_avg is not None:
                a.avg_2.append(fr.dist2_fused_avg)
            if fr.dist1_fused_wavg is not None:
                a.wavg_1.append(fr.dist1_fused_wavg)
            if fr.dist2_fused_wavg is not None:
                a.wavg_2.append(fr.dist2_fused_wavg)

            # Mean-between-UAVs per frame (kept for existing fused table)
            m_bi = mean2(fr.dist1_fused_bi, fr.dist2_fused_bi)
            if m_bi is not None:
                a.bi.append(m_bi)

            m_avg = mean2(fr.dist1_fused_avg, fr.dist2_fused_avg)
            if m_avg is not None:
                a.avg.append(m_avg)

            m_wavg = mean2(fr.dist1_fused_wavg, fr.dist2_fused_wavg)
            if m_wavg is not None:
                a.wavg.append(m_wavg)

            if fr.best_estimators or fr.best_fusions:
                a.frames_with_votes += 1
                add_votes(a.est_votes, fr.best_estimators, VOTE_MODE)
                add_votes(a.fus_votes, fr.best_fusions, VOTE_MODE)

    return agg


def aggregate_overall_by_angle(samples: List[SampleData]) -> Dict[int, Dict[str, float]]:
    per_angle = defaultdict(lambda: defaultdict(list))

    for s in samples:
        for fr in s.frames:
            gt = float(s.dist_gt)

            hb = mean2(fr.dist1_height, fr.dist2_height)
            if hb is not None:
                per_angle[s.angle]["height_based"].append((hb, gt))

            gb = mean2(fr.dist1_pitch, fr.dist2_pitch)
            if gb is not None:
                per_angle[s.angle]["ground_based"].append((gb, gt))

            inter = mean2(fr.dist1_fused_bi, fr.dist2_fused_bi)
            if inter is not None:
                per_angle[s.angle]["intersection"].append((inter, gt))

            av = mean2(fr.dist1_fused_avg, fr.dist2_fused_avg)
            if av is not None:
                per_angle[s.angle]["average"].append((av, gt))

            wavg = mean2(fr.dist1_fused_wavg, fr.dist2_fused_wavg)
            if wavg is not None:
                per_angle[s.angle]["weighted_avg"].append((wavg, gt))

    out: Dict[int, Dict[str, float]] = {}
    for angle, methods in per_angle.items():
        out[angle] = {}
        for method, pairs in methods.items():
            out[angle][method] = (
                float("nan")
                if not pairs
                else math.sqrt(sum((est - gt) ** 2 for est, gt in pairs) / len(pairs))
            )
    return out


# --------------------------
# LaTeX
# --------------------------

def latex_table_single_uav_per_uav(agg: Dict[ConfigKey, ConfigAgg]) -> str:
    lines = []
    lines.append(r"\\begin{table}[h]")
    lines.append(r"\\centering")
    lines.append(r"\\caption{Single-UAV distance estimation RMSE (meters) for ground-plane and height-based approaches.}")
    lines.append(r"\\label{tab:single_uav_results}")
    lines.append(r"\\begin{tabular}{c c c cc cc}")
    lines.append(r"\\toprule")
    lines.append(r"\\multirow{2}{*}{\\textbf{Angle}} &")
    lines.append(r"\\multirow{2}{*}{\\textbf{Distance}} &")
    lines.append(r"\\multirow{2}{*}{\\textbf{Height}} &")
    lines.append(r"\\multicolumn{2}{c}{\\textbf{Ground-based}} &")
    lines.append(r"\\multicolumn{2}{c}{\\textbf{Height-based}} \\\\")
    lines.append(r"\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    lines.append(r" & & & \\textbf{UAV 1} & \\textbf{UAV 2} & \\textbf{UAV 1} & \\textbf{UAV 2} \\\\")
    lines.append(r"\\midrule")

    for key in sorted(agg.keys(), key=lambda k: (k.angle, k.dist_gt, k.height_gt)):
        a = agg[key]
        gt = float(key.dist_gt)
        g1 = rmse(a.g1, gt)
        g2 = rmse(a.g2, gt)
        h1 = rmse(a.h1, gt)
        h2 = rmse(a.h2, gt)

        bold_g1 = bold_g2 = bold_h1 = bold_h2 = False
        if BOLD_RULE_SINGLE == "row_min":
            m = min(g1, g2, h1, h2)
            bold_g1, bold_g2, bold_h1, bold_h2 = (g1 == m), (g2 == m), (h1 == m), (h2 == m)
        elif BOLD_RULE_SINGLE == "group_min":
            mg = min(g1, g2)
            mh = min(h1, h2)
            bold_g1, bold_g2 = (g1 == mg), (g2 == mg)
            bold_h1, bold_h2 = (h1 == mh), (h2 == mh)
        else:
            raise ValueError("Unknown BOLD_RULE_SINGLE")

        lines.append(
            f"{key.angle}\\N{{\\circ}} & {key.dist_gt}\\,m & {key.height_gt}\\,m & "
            f"{fmt_rmse(g1, bold_g1)} & {fmt_rmse(g2, bold_g2)} & "
            f"{fmt_rmse(h1, bold_h1)} & {fmt_rmse(h2, bold_h2)} \\\\")

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    return "\n".join(lines)

def latex_table_single_uav_pooled(agg: Dict[ConfigKey, ConfigAgg]) -> str:
    """
    Table for single UAV estimation showing pooled RMSEs for ground-based and height-based methods.
    All estimates from UAV1 and UAV2 are pooled together for each config.
    """
    lines = []
    lines.append(r"\\begin{table}[h]")
    lines.append(r"\\centering")
    lines.append(r"\\caption{Single-UAV pooled RMSE (meters) for ground-based and height-based approaches.}")
    lines.append(r"\\label{tab:single_uav_pooled_results}")
    lines.append(r"\\begin{tabular}{c c c c c}")
    lines.append(r"\\toprule")
    lines.append(r"\\textbf{Angle} & \\textbf{Distance} & \\textbf{Height} & \\textbf{Ground-based} & \\textbf{Height-based} \\\\")
    lines.append(r"\\midrule")

    for key in sorted(agg.keys(), key=lambda k: (k.angle, k.dist_gt, k.height_gt)):
        a = agg[key]
        gt = float(key.dist_gt)
        pooled_g = a.g1 + a.g2
        pooled_h = a.h1 + a.h2
        rmse_g = rmse(pooled_g, gt)
        rmse_h = rmse(pooled_h, gt)

        # Bold the minimum between ground-based and height-based pooled RMSE
        m = min(rmse_g if rmse_g is not None else float('inf'),
                rmse_h if rmse_h is not None else float('inf'))
        bold_g = rmse_g == m if rmse_g is not None else False
        bold_h = rmse_h == m if rmse_h is not None else False

        lines.append(
            f"{key.angle}\\N{{\\circ}} & {key.dist_gt}\\,m & {key.height_gt}\\,m & "
            f"{fmt_rmse(rmse_g, bold_g)} & {fmt_rmse(rmse_h, bold_h)} \\\\"
        )

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    return "\n".join(lines)

def latex_table_dual_fused_pooled(agg: Dict[ConfigKey, ConfigAgg], include_weighted: bool = True) -> str:
    """
    Table for dual-UAV fused localization showing pooled RMSEs for intersection, average, and weighted average methods.
    All fused estimates from UAV1 and UAV2 are pooled together for each config.
    """
    cols = r"c c c c c" if not include_weighted else r"c c c c c c"
    lines = []
    lines.append(r"\\begin{table}[h]")
    lines.append(r"\\centering")
    lines.append(r"\\caption{Dual-UAV pooled RMSE (meters) for fused localization (intersection, average, weighted average).}")
    lines.append(r"\\label{tab:fused_pooled_results}")
    lines.append(rf"\\begin{{tabular}}{{{cols}}}")
    lines.append(r"\\toprule")
    if include_weighted:
        lines.append(r"\\textbf{Angle} & \\textbf{Distance} & \\textbf{Height} & \\textbf{Intersection} & \\textbf{Average} & \\textbf{Weighted Avg} \\\\")
    else:
        lines.append(r"\\textbf{Angle} & \\textbf{Distance} & \\textbf{Height} & \\textbf{Intersection} & \\textbf{Average} \\\\")
    lines.append(r"\\midrule")

    for key in sorted(agg.keys(), key=lambda k: (k.angle, k.dist_gt, k.height_gt)):
        a = agg[key]
        gt = float(key.dist_gt)
        pooled_bi = a.bi_1 + a.bi_2
        pooled_avg = a.avg_1 + a.avg_2
        pooled_wavg = a.wavg_1 + a.wavg_2 if include_weighted else None
        rmse_bi = rmse(pooled_bi, gt)
        rmse_avg = rmse(pooled_avg, gt)
        rmse_wavg = rmse(pooled_wavg, gt) if include_weighted else None

        # Bold the minimum among the pooled RMSEs
        vals = [v for v in [rmse_bi, rmse_avg, rmse_wavg] if v is not None]
        m = min(vals) if vals else float('inf')
        bold_bi = rmse_bi == m
        bold_avg = rmse_avg == m
        bold_wavg = rmse_wavg == m if include_weighted else False

        if include_weighted:
            lines.append(
                f"{key.angle}\\N{{\\circ}} & {key.dist_gt}\\,m & {key.height_gt}\\,m & "
                f"{fmt_rmse(rmse_bi, bold_bi)} & {fmt_rmse(rmse_avg, bold_avg)} & {fmt_rmse(rmse_wavg, bold_wavg)} \\\\"
            )
        else:
            lines.append(
                f"{key.angle}\\N{{\\circ}} & {key.dist_gt}\\,m & {key.height_gt}\\,m & "
                f"{fmt_rmse(rmse_bi, bold_bi)} & {fmt_rmse(rmse_avg, bold_avg)} \\\\"
            )

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    return "\n".join(lines)

def latex_table_single_uav(agg: Dict[ConfigKey, ConfigAgg]) -> str:
    """
    Table for single UAV estimation showing combined (mean of UAV1 and UAV2) RMSEs for ground-based and height-based methods.
    Columns: Angle, Distance, Height, Ground-based Combined, Height-based Combined
    """
    lines = []
    lines.append(r"\\begin{table}[h]")
    lines.append(r"\\centering")
    lines.append(r"\\caption{Single-UAV combined RMSE (meters) for ground-based and height-based approaches.}")
    lines.append(r"\\label{tab:single_uav_combined_results}")
    lines.append(r"\\begin{tabular}{c c c c c}")
    lines.append(r"\\toprule")
    lines.append(r"\\textbf{Angle} & \\textbf{Distance} & \\textbf{Height} & \\textbf{Ground-based} & \\textbf{Height-based} \\\\")
    lines.append(r"\\midrule")

    for key in sorted(agg.keys(), key=lambda k: (k.angle, k.dist_gt, k.height_gt)):
        a = agg[key]
        gt = float(key.dist_gt)
        g1 = rmse(a.g1, gt)
        g2 = rmse(a.g2, gt)
        h1 = rmse(a.h1, gt)
        h2 = rmse(a.h2, gt)
        g_combined = mean2(g1, g2)
        h_combined = mean2(h1, h2)

        # Bold the minimum between ground-based and height-based combined
        m = min(g_combined if g_combined is not None else float('inf'),
                h_combined if h_combined is not None else float('inf'))
        bold_g_combined = g_combined == m if g_combined is not None else False
        bold_h_combined = h_combined == m if h_combined is not None else False

        lines.append(
            f"{key.angle}\\N{{\\circ}} & {key.dist_gt}\\,m & {key.height_gt}\\,m & "
            f"{fmt_rmse(g_combined, bold_g_combined)} & {fmt_rmse(h_combined, bold_h_combined)} \\\\"
        )

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    return "\n".join(lines)

def latex_table_dual_fused(agg: Dict[ConfigKey, ConfigAgg], include_weighted: bool = True) -> str:
    cols = r"c c c c c" if not include_weighted else r"c c c c c c"
    lines = []
    lines.append(r"\\begin{table}[h]")
    lines.append(r"\\centering")
    lines.append(r"\\caption{Dual-UAV fused localization RMSE (meters) using ray-intersection triangulation and coordinate averaging.}")
    lines.append(r"\\label{tab:fused_results}")
    lines.append(rf"\\begin{{tabular}}{{{cols}}}")
    lines.append(r"\\toprule")
    if include_weighted:
        lines.append(r"\\textbf{Angle} & \\textbf{Distance} & \\textbf{Height} & \\textbf{Intersection} & \\textbf{Average} & \\textbf{Weighted Avg} \\\\")
    else:
        lines.append(r"\\textbf{Angle} & \\textbf{Distance} & \\textbf{Height} & \\textbf{Intersection} & \\textbf{Average} \\\\")
    lines.append(r"\\midrule")

    for key in sorted(agg.keys(), key=lambda k: (k.angle, k.dist_gt, k.height_gt)):
        a = agg[key]
        gt = float(key.dist_gt)

        r_bi = rmse(a.bi, gt)
        r_avg = rmse(a.avg, gt)
        r_w = rmse(a.wavg, gt) if include_weighted else None

        bold_bi = bold_avg = bold_w = False
        if BOLD_RULE_FUSION == "row_min":
            if include_weighted:
                m = min(r_bi, r_avg, r_w)
                bold_bi, bold_avg, bold_w = (r_bi == m), (r_avg == m), (r_w == m)
            else:
                m = min(r_bi, r_avg)
                bold_bi, bold_avg = (r_bi == m), (r_avg == m)
        else:
            raise ValueError("Unknown BOLD_RULE_FUSION")

        if include_weighted:
            lines.append(
                f"{key.angle}\\N{{\\circ}} & {key.dist_gt}\\,m & {key.height_gt}\\,m & "
                f"{fmt_rmse(r_bi, bold_bi)} & {fmt_rmse(r_avg, bold_avg)} & {fmt_rmse(r_w, bold_w)} \\\\")
        else:
            lines.append(
                f"{key.angle}\\N{{\\circ}} & {key.dist_gt}\\,m & {key.height_gt}\\,m & "
                f"{fmt_rmse(r_bi, bold_bi)} & {fmt_rmse(r_avg, bold_avg)} \\\\")

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    return "\n".join(lines)


def latex_table_fused_per_uav(agg: Dict[ConfigKey, ConfigAgg]) -> str:
    """Per-UAV RMSE for fused localization (Intersection + Average).

    This matches the 2x2 header structure you posted:
      - Intersection: UAV1, UAV2
      - Average:      UAV1, UAV2

    Bolding rule: within each row and method, bold the smaller RMSE between UAV1 and UAV2.
    """

    lines: List[str] = []
    lines.append(r"\\begin{table}[t]")
    lines.append(r"\\centering")
    lines.append(r"\\caption{Fused Localization RMSE (meters).}")
    lines.append(r"\\label{tab:fused_results_per_uav}")
    lines.append(r"\\begin{tabular}{c c c cc cc}")
    lines.append(r"\\toprule")
    lines.append(r"\\multirow{2}{*}{\\textbf{Angle}} & ")
    lines.append(r"\\multirow{2}{*}{\\textbf{Distance}} & ")
    lines.append(r"\\multirow{2}{*}{\\textbf{Height}} &")
    lines.append(r"\\multicolumn{2}{c}{\\textbf{Intersection}} &")
    lines.append(r"\\multicolumn{2}{c}{\\textbf{Average}} \\\\")
    lines.append(r"\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    lines.append(r" & & & \\textbf{UAV1} & \\textbf{UAV2} & \\textbf{UAV1} & \\textbf{UAV2} \\\\")
    lines.append(r"\\midrule")

    keys_sorted = sorted(agg.keys(), key=lambda k: (k.angle, k.dist_gt, k.height_gt))
    last_angle: Optional[int] = None
    for key in keys_sorted:
        if last_angle is not None and key.angle != last_angle:
            lines.append(r"\\midrule")
        last_angle = key.angle

        a = agg[key]
        gt = float(key.dist_gt)

        r_bi_1 = rmse(a.bi_1, gt)
        r_bi_2 = rmse(a.bi_2, gt)
        r_avg_1 = rmse(a.avg_1, gt)
        r_avg_2 = rmse(a.avg_2, gt)

        bi_min = min(r_bi_1, r_bi_2)
        avg_min = min(r_avg_1, r_avg_2)

        lines.append(
            f"{key.angle} & {key.dist_gt} & {key.height_gt} & "
            f"{fmt_rmse(r_bi_1, bold=(r_bi_1 == bi_min))} & {fmt_rmse(r_bi_2, bold=(r_bi_2 == bi_min))} & "
            f"{fmt_rmse(r_avg_1, bold=(r_avg_1 == avg_min))} & {fmt_rmse(r_avg_2, bold=(r_avg_2 == avg_min))} \\\\")

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    return "\n".join(lines)


def latex_table_overall_by_angle(overall: Dict[int, Dict[str, float]]) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall localization RMSE comparison across estimation and fusion strategies (meters).}")
    lines.append(r"\label{tab:rmse_final_comparison_by_angle}")
    lines.append(r"\begin{tabular}{c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Angle} & \textbf{Height-based} & \textbf{Ground-based} & \textbf{Intersection} & \textbf{Average} & \textbf{Weighted Avg} \\")
    lines.append(r"\midrule")

    for angle in sorted(overall.keys()):
        hb = overall[angle].get("height_based", float("nan"))
        gb = overall[angle].get("ground_based", float("nan"))
        inter = overall[angle].get("intersection", float("nan"))
        av = overall[angle].get("average", float("nan"))
        wavg = overall[angle].get("weighted_avg", float("nan"))

        m = min(hb, gb, inter, av, wavg)

        lines.append(
            f"{angle}° & "
            f"{fmt_rmse(hb, hb==m)} & "
            f"{fmt_rmse(gb, gb==m)} & "
            f"{fmt_rmse(inter, inter==m)} & "
            f"{fmt_rmse(av, av==m)} & "
            f"{fmt_rmse(wavg, wavg==m)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_table_votes(agg: Dict[ConfigKey, ConfigAgg], kind: str = "estimator", top_k: int = 3) -> str:
    assert kind in ("estimator", "fusion")
    caption = (
        "Per-configuration winner frequency from the log (best estimator)."
        if kind == "estimator"
        else "Per-configuration winner frequency from the log (best fusion)."
    )
    label = "tab:best_estimator_votes" if kind == "estimator" else "tab:best_fusion_votes"

    lines = []
    lines.append(r"\\begin{table}[h]")
    lines.append(r"\\centering")
    lines.append(rf"\\caption{{{caption}}}")
    lines.append(rf"\\label{{{label}}}")
    lines.append(r"\\begin{tabular}{c c c l}")
    lines.append(r"\\toprule")
    lines.append(r"\\textbf{Angle} & \\textbf{Distance} & \\textbf{Height} & \\textbf{Top winners (votes)} \\\\")
    lines.append(r"\\midrule")

    for key in sorted(agg.keys(), key=lambda k: (k.angle, k.dist_gt, k.height_gt)):
        a = agg[key]
        counter = a.est_votes if kind == "estimator" else a.fus_votes
        tops = topk(counter, k=top_k)
        s = ", ".join([f"{name} ({votes:.2f})" for name, votes in tops]) if tops else "--"
        lines.append(f"{key.angle}\\N{{\\circ}} & {key.dist_gt}\\,m & {key.height_gt}\\,m & {s} \\\\")

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    return "\n".join(lines)


def write_votes_summary_txt(agg: Dict[ConfigKey, ConfigAgg], outpath: str):
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(f"Vote mode: {VOTE_MODE}\n\n")
        for key in sorted(agg.keys(), key=lambda k: (k.angle, k.dist_gt, k.height_gt)):
            a = agg[key]
            f.write(f"Config (angle={key.angle}°, dist={key.dist_gt}m, height={key.height_gt}m)\n")
            f.write(f"  frames_with_votes: {a.frames_with_votes}\n")
            f.write("  best estimator votes:\n")
            for name, v in sorted(a.est_votes.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"    - {name}: {v:.3f}\n")
            f.write("  best fusion votes:\n")
            for name, v in sorted(a.fus_votes.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"    - {name}: {v:.3f}\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no_weighted", action="store_true")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    samples = parse_log(args.log)
    agg = aggregate(samples)
    overall = aggregate_overall_by_angle(samples)

    tables = []
    tables.append(latex_table_single_uav(agg))
    tables.append("")
    tables.append(latex_table_single_uav_per_uav(agg))
    tables.append("")
    tables.append(latex_table_single_uav_pooled(agg))
    tables.append("")
    tables.append(latex_table_dual_fused(agg, include_weighted=(not args.no_weighted)))
    tables.append("")
    tables.append(latex_table_fused_per_uav(agg))
    tables.append("")
    tables.append(latex_table_dual_fused_pooled(agg, include_weighted=(not args.no_weighted)))
    tables.append("")
    tables.append(latex_table_overall_by_angle(overall))
    tables.append("")
    tables.append(latex_table_votes(agg, kind="estimator", top_k=args.topk))
    tables.append("")
    tables.append(latex_table_votes(agg, kind="fusion", top_k=args.topk))

    tex_path = os.path.join(args.outdir, "tables.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(tables))

    votes_path = os.path.join(args.outdir, "votes_summary.txt")
    write_votes_summary_txt(agg, votes_path)

    print(f"Wrote LaTeX tables: {tex_path}")
    print(f"Wrote votes summary: {votes_path}")


if __name__ == "__main__":
    main()
