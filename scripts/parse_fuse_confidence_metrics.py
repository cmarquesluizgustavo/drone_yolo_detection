import re
from collections import defaultdict, deque

LOG_PATH = "logs/45.log"
THRESHOLDS = [0.3, 0.5, 0.7]
labels = [
    ("drone1", "Drone 1"),
    ("drone2", "Drone 2"),
    ("metodo1", "Fused"),   # FUSED = method 1
    ("metodo2", "MEAN"),    # MEAN = method 2
    ("metodo3", "MAX")      # MAX = method 3
]
WINDOW = 10
VOTE = 5

frame_results = {th: {} for th in THRESHOLDS}
sample_results = {th: {} for th in THRESHOLDS}

def calc(acc, prec, rec):
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1

def format_metric_row(label, acc, prec, rec, f1):
    return f"{label} & {acc:.3f} & {prec:.3f} & {rec:.3f} & {f1:.3f} \\"  # no newline

with open(LOG_PATH, encoding="utf-8") as f:
    lines = f.readlines()

for THRESHOLD in THRESHOLDS:
    metrics = {k: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for k, _ in labels}
    temporal_results = {k: defaultdict(list) for k, _ in labels}
    gt = None
    current_sample = None
    current_frame = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("Processing sample"):
            m = re.search(r":\s*([a-zA-Z]+)_", line)
            if m:
                gt = True if m.group(1).lower() == "real" else False
            else:
                gt = None
            current_sample = line.strip().split(":",1)[-1].strip()
            current_frame = -1
        if "Frame" in line and ":" in line:
            try:
                current_frame = int(re.search(r"Frame (\d+):", line).group(1))
            except Exception:
                current_frame = -1
        if line.strip().startswith("DETECTION:") and gt is not None:
            m = re.search(r"w1=([0-9.]+), w2=([0-9.]+), w_fused=([0-9.]+)", line)
            if m:
                d1_conf = float(m.group(1))
                d2_conf = float(m.group(2))
                fused_conf = float(m.group(3))
                # For mean and max, need to recompute
                mean_conf = (d1_conf + d2_conf) / 2.0
                max_conf = max(d1_conf, d2_conf)
                all_confs = [d1_conf, d2_conf, fused_conf, mean_conf, max_conf]
                for idx, (key, _) in enumerate(labels):
                    conf = all_confs[idx]
                    detected = conf >= THRESHOLD
                    # Per-frame
                    if detected and gt:
                        metrics[key]["tp"] += 1
                    elif detected and not gt:
                        metrics[key]["fp"] += 1
                    elif not detected and not gt:
                        metrics[key]["tn"] += 1
                    elif not detected and gt:
                        metrics[key]["fn"] += 1
                    # For temporal voting
                    temporal_results[key][current_sample].append((current_frame, detected, gt))
    # Store per-frame metrics
    for key, label in labels:
        tp = metrics[key]["tp"]
        fp = metrics[key]["fp"]
        tn = metrics[key]["tn"]
        fn = metrics[key]["fn"]
        total_cases = tp + fp + tn + fn
        acc = (tp + tn) / total_cases if total_cases else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = calc(acc, prec, rec)
        frame_results[THRESHOLD][label] = (acc, prec, rec, f1)
    # Temporal voting metrics
    for key, label in labels:
        t_tp = t_fp = t_tn = t_fn = 0
        for sample, results in temporal_results[key].items():
            results_sorted = sorted(results, key=lambda x: x[0])
            detected_windows = []
            gt_val = None
            dq = deque()
            for frame, detected, gt in results_sorted:
                dq.append(detected)
                if len(dq) > WINDOW:
                    dq.popleft()
                if len(dq) == WINDOW:
                    window_detected = sum(dq) >= VOTE
                    detected_windows.append(window_detected)
                gt_val = gt
            for detected in detected_windows:
                if detected and gt_val:
                    t_tp += 1
                elif detected and not gt_val:
                    t_fp += 1
                elif not detected and not gt_val:
                    t_tn += 1
                elif not detected and gt_val:
                    t_fn += 1
        t_total = t_tp + t_fp + t_tn + t_fn
        t_acc = (t_tp + t_tn) / t_total if t_total else 0
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) else 0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) else 0
        t_f1 = calc(t_acc, t_prec, t_rec)
        sample_results[THRESHOLD][label] = (t_acc, t_prec, t_rec, t_f1)

# Print LaTeX tables
for THRESHOLD in THRESHOLDS:
    print("% =========================")
    print(f"% Detection – frame-level (threshold {THRESHOLD})")
    print("% =========================")
    print("\\begin{table}[ht]")
    print("\\centering")
    print(f"\\caption{{Detection, threshold {THRESHOLD}}}")
    print(f"\\label{{tab:detection_frame_metrics_{str(THRESHOLD).replace('.', '')}}}")
    print("\\begin{tabular}{l c c c c}")
    print("\\toprule")
    print("\\textbf{Setup} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} \\")
    print("\\midrule")
    for _, label in labels:
        acc, prec, rec, f1 = frame_results[THRESHOLD][label]
        print(format_metric_row(label, acc, prec, rec, f1))
    print("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("% =========================")
    print(f"% Detection – sample-level (majority voting 5/10, threshold {THRESHOLD})")
    print("% =========================")
    print("\\begin{table}[ht]")
    print("\\centering")
    print(f"\\caption{{Detection (majority voting 5/10), threshold {THRESHOLD}}}")
    print(f"\\label{{tab:detection_sample_metrics_{str(THRESHOLD).replace('.', '')}}}")
    print("\\begin{tabular}{l c c c c}")
    print("\\toprule")
    print("\\textbf{Setup} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} \\")
    print("\\midrule")
    for _, label in labels:
        acc, prec, rec, f1 = sample_results[THRESHOLD][label]
        print(format_metric_row(label, acc, prec, rec, f1))
    print("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
