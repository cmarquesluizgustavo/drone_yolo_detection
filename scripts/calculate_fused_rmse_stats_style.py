
import re
import math
from collections import defaultdict

# Path to the log file
LOG_PATH = "logs/run_20260219_161601_console.log"

# Regular expressions
angle_re = re.compile(r"PROCESSING ANGLE: (\d+)°")
sample_re = re.compile(r"Processing sample \d+/\d+: (\w+)_([\d]+)_([\d]+)_clip_\d+_1080p_none_10fps_every10frames")
fused_avg_re = re.compile(r"FUSED_AVERAGE: dist1_from_fused_avg=([\d.]+), dist2_from_fused_avg=([\d.]+)")
fused_bi_re = re.compile(r"FUSED_BEARING_INTERSECTION: dist1_from_fused_bi=([\d.]+), dist2_from_fused_bi=([\d.]+)")

# Data structure: {(method, d_source, gt_distance, height, angle): [(est, real)]}
data = defaultdict(list)

# Add regex for monocular methods
pinhole_re = re.compile(r"dist1_height-based=([\d.]+).*dist2_height-based=([\d.]+)")
pitch_re = re.compile(r"dist1_pitch-based=([\d.]+).*dist2_pitch-based=([\d.]+)")

with open(LOG_PATH, encoding="utf-8") as f:
    angle = None
    gt_distance = None
    height = None
    for line in f:
        angle_match = angle_re.search(line)
        if angle_match:
            angle = int(angle_match.group(1))
            continue
        sample_match = sample_re.search(line)
        if sample_match:
            gt_distance = float(sample_match.group(2))
            height = float(sample_match.group(3))
            continue
        # Monocular methods
        pinhole_match = pinhole_re.search(line)
        if pinhole_match and gt_distance is not None and angle is not None and height is not None:
            est1 = float(pinhole_match.group(1))
            est2 = float(pinhole_match.group(2))
            data[("pinhole", "d1", gt_distance, height, angle)].append((est1, gt_distance))
            data[("pinhole", "d2", gt_distance, height, angle)].append((est2, gt_distance))
            data[("pinhole", "all", gt_distance, height, angle)].append(((est1+est2)/2, gt_distance))
        pitch_match = pitch_re.search(line)
        if pitch_match and gt_distance is not None and angle is not None and height is not None:
            est1 = float(pitch_match.group(1))
            est2 = float(pitch_match.group(2))
            data[("pitch", "d1", gt_distance, height, angle)].append((est1, gt_distance))
            data[("pitch", "d2", gt_distance, height, angle)].append((est2, gt_distance))
            data[("pitch", "all", gt_distance, height, angle)].append(((est1+est2)/2, gt_distance))
        # Fused methods
        avg_match = fused_avg_re.search(line)
        if avg_match and gt_distance is not None and angle is not None and height is not None:
            est1 = float(avg_match.group(1))
            est2 = float(avg_match.group(2))
            data[("avg", "d1", gt_distance, height, angle)].append((est1, gt_distance))
            data[("avg", "d2", gt_distance, height, angle)].append((est2, gt_distance))
            data[("avg", "all", gt_distance, height, angle)].append(((est1+est2)/2, gt_distance))
            continue
        bi_match = fused_bi_re.search(line)
        if bi_match and gt_distance is not None and angle is not None and height is not None:
            est1 = float(bi_match.group(1))
            est2 = float(bi_match.group(2))
            data[("bi", "d1", gt_distance, height, angle)].append((est1, gt_distance))
            data[("bi", "d2", gt_distance, height, angle)].append((est2, gt_distance))
            data[("bi", "all", gt_distance, height, angle)].append(((est1+est2)/2, gt_distance))
            continue

def rmse(pairs):
    if not pairs:
        return float('nan')
    return math.sqrt(sum((est-real)**2 for est, real in pairs) / len(pairs))




angles = sorted({k[4] for k in data.keys()})
fusion_types = ["avg", "bi"]
drones = ["d1", "d2"]
gt_distances = sorted({k[2] for k in data.keys()})
heights = sorted({k[3] for k in data.keys()})







def latex_table(method, caption, label):
    print("% =========================")
    print(f"% {caption}")
    print("% =========================")
    print("\\begin{table}[ht]")
    print("\\centering")
    print(f"\\caption{{{caption} (meters)}}")
    print(f"\\label{{{label}}}")
    print("\\begin{tabular}{c c c c c}")
    print("\\toprule")
    print("\\textbf{Angle} & \\textbf{Dist.} & \\textbf{Height} & \\textbf{UAV 1} & \\textbf{UAV 2} \\")
    print("\\midrule")
    for ang in angles:
        for gt in gt_distances:
            for h in heights:
                row = []
                for drone in drones:
                    pairs = data.get((method, drone, gt, h, ang), [])
                    if pairs:
                        row.append(f"{rmse(pairs):.3f}")
                    else:
                        row.append("--")
                # Bold the row for (gt==5.0, h==5.0) as in example
                if gt == 5.0 and h == 5.0:
                    print(f"\\textbf{{{ang}}} & \\textbf{{{gt}}} & \\textbf{{{h}}} & \\textbf{{{row[0]}}} & \\textbf{{{row[1]}}} \\")
                else:
                    print(f"{ang} & {gt} & {h} & {row[0]} & {row[1]} \\")
            if gt == 10.0 and h == 5.0:
                print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

# Output LaTeX tables for each method
latex_table("pitch", "Ground-Plane-based RMSE", "tab:rmse_pitch_by_combo")
latex_table("pinhole", "Height-based RMSE", "tab:rmse_Height-based_by_combo")
latex_table("bi", "Fused RMSE - Bearing Intersection", "tab:rmse_fused_geo_by_combo_bi")
latex_table("avg", "Fused RMSE - Coordinate Averaging", "tab:rmse_fused_geo_by_combo_avg")

# Overall RMSE comparison tables
def overall_latex_table(methods, captions, label):
    print("% =========================")
    print(f"% {captions}")
    print("% =========================")
    print("\\begin{table}[ht]")
    print("\\centering")
    print(f"\\caption{{{captions} (meters)}}")
    print(f"\\label{{{label}}}")
    n = len(methods)
    coldef = 'c ' + ' '.join(['c c']*n)
    print(f"\\begin{{tabular}}{{{coldef}}}")
    print("\\toprule")
    header = [f"\\multicolumn{{2}}{{c}}{{\\textbf{{{m}}}}}" for m in methods]
    print("\\textbf{Angle} & " + " & ".join(header) + " \\")
    cmid = []
    for i in range(n):
        start = 2 + i*2
        end = start+1
        cmid.append(f"\\cmidrule(lr){{{start}-{end}}}")
    print(" ".join(cmid))
    print("& " + " & ".join(["\\textbf{UAV 1} & \\textbf{UAV 2}"]*n) + r"\\")
    print("\\midrule")
    for ang in angles:
        row = [str(ang)]
        for method in methods:
            for drone in drones:
                # Aggregate all (gt, h) for this angle
                pairs = []
                for gt in gt_distances:
                    for h in heights:
                        pairs.extend(data.get((method, drone, gt, h, ang), []))
                if pairs:
                    row.append(f"{rmse(pairs):.3f}")
                else:
                    row.append("--")
        print(" & ".join(row) + r"\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

# Monocular overall
overall_latex_table(["pitch", "pinhole"], "Overall RMSE Comparison", "tab:rmse_overall_comparison_mono")

# Fused overall
overall_latex_table(["bi", "avg"], "Overall Fused RMSE Comparison", "tab:rmse_overall_comparison_fused")

# Final table: compare all methods by angle only (average over all distances, heights, drones)
def final_comparison_by_angle():
    print("% =========================")
    print("% Final Overall RMSE by Method and Angle")
    print("% =========================")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Final Overall RMSE by Method and Angle (meters)}")
    print("\\label{tab:rmse_final_comparison_by_angle}")
    print("\\begin{tabular}{c c c c c}")
    print("\\toprule")
    print("\\textbf{Angle} & \\textbf{Pinhole} & \\textbf{Pitch} & \\textbf{Avg} & \\textbf{BI} \\")
    print("\\midrule")
    methods = ["pinhole", "pitch", "avg", "bi"]
    for ang in angles:
        row = [str(ang)]
        for method in methods:
            pairs = []
            for drone in drones:
                for gt in gt_distances:
                    for h in heights:
                        pairs.extend(data.get((method, drone, gt, h, ang), []))
            if pairs:
                row.append(f"{rmse(pairs):.3f}")
            else:
                row.append("--")
        print(" & ".join(row) + r"\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

final_comparison_by_angle()
