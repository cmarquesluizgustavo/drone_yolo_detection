import math
from collections import defaultdict


class DetectionStatistics:
    """
    Comprehensive detection + distance evaluation container.

    Responsibilities:
    - Frame-level weapon detection metrics
    - Sample-level majority voting metrics
    - Segmented metrics (distance / height / class)
    - Unified RMSE logging for pinhole / pitch / fused
    """

    # -------------------------
    # INIT / RESET / LIFECYCLE
    # -------------------------

    def __init__(self, sample_majority_threshold=1, min_rmse_samples=1):
        self.sample_majority_threshold = sample_majority_threshold
        self.min_rmse_samples = min_rmse_samples
        self.reset()

    def reset(self):
        # --- lifecycle ---
        self.in_sample = False

        # --- image / people ---
        self.total_images = 0
        self.images_with_people = 0
        self.total_people = 0
        self.total_weapons = 0
        self.people_with_weapons = 0

        # --- samples ---
        self.total_samples = 0
        self.samples_with_weapons = 0

        # --- frame confusion ---
        self.tp_frame = 0
        self.tn_frame = 0
        self.fp_frame = 0
        self.fn_frame = 0

        # --- sample confusion ---
        self.tp_sample = 0
        self.tn_sample = 0
        self.fp_sample = 0
        self.fn_sample = 0

        self.sample_metrics_by_class = {}

        # --- segmented detection metrics ---
        self.metrics_by_distance = {}
        self.metrics_by_height = {}
        self.metrics_by_class = {}

        # --- distance stats ---
        self.distances = []
        self.people_with_distance = 0

        # --- unified RMSE storage ---
        self.rmse_pairs = []  # canonical store

        # --- current sample tracking ---
        self.current_sample_ground_truth = False
        self.current_sample_class = None
        self.current_sample_frames_with_weapons = 0
        self.current_sample_total_frames = 0

    # -------------------------
    # BUCKETING HELPERS
    # -------------------------

    def bucket_distance(self, d):
        return int(round(d))

    def bucket_pitch(self, pitch_real, pitch_annot):
        return pitch_real if pitch_real is not None else pitch_annot

    # -------------------------
    # SAMPLE HANDLING
    # -------------------------

    def start_new_sample(self, sample_ground_truth=False, sample_class=None):
        if self.in_sample:
            self.finalize_current_sample()

        self.in_sample = True
        self.total_samples += 1

        self.current_sample_ground_truth = sample_ground_truth
        self.current_sample_class = sample_class
        self.current_sample_frames_with_weapons = 0
        self.current_sample_total_frames = 0

    def finalize_current_sample(self):
        if not self.in_sample:
            return

        detected = (
            self.current_sample_frames_with_weapons
            >= self.sample_majority_threshold
        )

        if detected:
            self.samples_with_weapons += 1

        if detected:
            if self.current_sample_ground_truth:
                result = "tp"
                self.tp_sample += 1
            else:
                result = "fp"
                self.fp_sample += 1
        else:
            if self.current_sample_ground_truth:
                result = "fn"
                self.fn_sample += 1
            else:
                result = "tn"
                self.tn_sample += 1

        if self.current_sample_class is not None:
            self.sample_metrics_by_class.setdefault(
                self.current_sample_class,
                {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
            )
            self.sample_metrics_by_class[self.current_sample_class][result] += 1

        self.in_sample = False

    # -------------------------
    # IMAGE RESULTS
    # -------------------------

    def add_image_results(
        self,
        num_people,
        num_weapons,
        people_with_weapons_count,
        has_weapon_ground_truth,
        distances=None,
        real_distance=None,
        cam_height_m=None,
        sample_class=None,
        camera_pitch_annotated_deg=None,
        camera_pitch_real_deg=None,
        distance_estimates=None,
    ):
        """
        distance_estimates: list of dicts like:
        {
            "est": float,
            "method": "pinhole" | "pitch" | "fused",
            "fusion_type": "avg" | "bi" | None,
            "d_source": "d1" | "d2" | "all" | None,
        }
        """

        # --- image stats ---
        self.total_images += 1
        if num_people > 0:
            self.images_with_people += 1

        self.total_people += num_people
        self.total_weapons += num_weapons
        self.people_with_weapons += people_with_weapons_count

        # --- sample tracking ---
        self.current_sample_total_frames += 1
        if num_weapons > 0:
            self.current_sample_frames_with_weapons += 1

        # --- frame confusion ---
        detected = num_weapons > 0

        if detected:
            if has_weapon_ground_truth:
                result = "tp"
                self.tp_frame += 1
            else:
                result = "fp"
                self.fp_frame += 1
        else:
            if has_weapon_ground_truth:
                result = "fn"
                self.fn_frame += 1
            else:
                result = "tn"
                self.tn_frame += 1

        # --- segmented detection metrics ---
        if real_distance is not None:
            d = self.bucket_distance(real_distance)
            self.metrics_by_distance.setdefault(
                d, {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            )
            self.metrics_by_distance[d][result] += 1

        if cam_height_m is not None:
            self.metrics_by_height.setdefault(
                cam_height_m, {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            )
            self.metrics_by_height[cam_height_m][result] += 1

        if sample_class is not None:
            self.metrics_by_class.setdefault(
                sample_class, {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            )
            self.metrics_by_class[sample_class][result] += 1

        # --- distance collection ---
        if distances:
            self.distances.extend(distances)
            self.people_with_distance += len(distances)

        # --- unified RMSE logging (FRAME-LEVEL GUARANTEE) ---
        if real_distance is not None and distance_estimates:
            dist_bucket = self.bucket_distance(real_distance)
            pitch_bucket = self.bucket_pitch(
                camera_pitch_real_deg, camera_pitch_annotated_deg
            )

            seen = set()  # (method, fusion_type, d_source)

            for d in distance_estimates:
                est = d.get("est")
                if est is None:
                    continue

                key = (
                    d.get("method"),
                    d.get("fusion_type"),
                    d.get("d_source"),
                )

                # Enforce ONE RMSE ENTRY per frame per method/fusion/source
                if key in seen:
                    continue
                seen.add(key)

                self.rmse_pairs.append(
                    {
                        "est": est,
                        "real": real_distance,
                        "method": d.get("method"),
                        "fusion_type": d.get("fusion_type"),
                        "d_source": d.get("d_source"),
                        "class": sample_class,
                        "distance": dist_bucket,
                        "height": cam_height_m,
                        "pitch": pitch_bucket,
                    }
                )

    # -------------------------
    # METRICS
    # -------------------------

    def calculate_metrics(self, tp, tn, fp, fn):
        total = tp + tn + fp + fn
        if total == 0:
            return 0, 0, 0, 0

        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * prec * rec / (prec + rec)
            if (prec + rec) > 0
            else 0
        )
        return acc, prec, rec, f1

    # -------------------------
    # RMSE
    # -------------------------

    def compute_rmse_filtered(self, **filters):
        pairs = [
            p
            for p in self.rmse_pairs
            if all(p.get(k) == v for k, v in filters.items())
            and p["est"] is not None
            and p["real"] is not None
        ]

        if len(pairs) < self.min_rmse_samples:
            return None

        mse = sum((p["est"] - p["real"]) ** 2 for p in pairs) / len(
            pairs
        )
        return math.sqrt(mse)

    # -------------------------
    # FINALIZE
    # -------------------------

    def finalize(self):
        if self.in_sample:
            self.finalize_current_sample()

    # -------------------------
    # SUMMARY PRINT
    # -------------------------

    def print_summary(self):
        print("\n" + "=" * 60)
        print("DETECTION SUMMARY")
        print("=" * 60)

        acc, prec, rec, f1 = self.calculate_metrics(
            self.tp_frame, self.tn_frame, self.fp_frame, self.fn_frame
        )

        print(f"Frames: {self.total_images}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1: {f1:.3f}")

        print("\nSAMPLE METRICS")
        acc, prec, rec, f1 = self.calculate_metrics(
            self.tp_sample, self.tn_sample, self.fp_sample, self.fn_sample
        )
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1: {f1:.3f}")

        print("\nRMSE SUMMARY")
        print("-" * 60)

        # --- Monocular ---
        for method in ["pinhole", "pitch"]:
            rmse = self.compute_rmse_filtered(method=method)
            if rmse is not None:
                print(f"{method.upper():<10}: {rmse:.3f} m")

        # --- Fused (all metrics, all confidence fusion methods) ---
        print("\nFUSED METRICS COMPARISON (confidence fusion methods)")
        fusion_methods = [
            (None, "Método 1: 1-(1-c1)*(1-c2)"),
            ("mean", "Método 2: Média"),
            ("max", "Método 3: Máximo")
        ]
        for fusion_method, label in fusion_methods:
            # Gather confusion counts for this method
            tp = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_method") == fusion_method and p.get("detected") and p.get("ground_truth"))
            fp = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_method") == fusion_method and p.get("detected") and not p.get("ground_truth"))
            tn = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_method") == fusion_method and not p.get("detected") and not p.get("ground_truth"))
            fn = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_method") == fusion_method and not p.get("detected") and p.get("ground_truth"))
            acc, prec, rec, f1 = self.calculate_metrics(tp, tn, fp, fn)
            rmse_fused = self.compute_rmse_filtered(method="fused", fusion_method=fusion_method)
            print(f"{label:<30}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} RMSE={rmse_fused if rmse_fused is not None else 'N/A'}")

        # --- Fused breakdown by type and method ---
        print("\nFUSED METRICS BREAKDOWN (por método de confiança)")
        for fusion_type in ["avg", "bi"]:
            for fusion_method, label in fusion_methods:
                # Gather confusion counts for this method/type
                tp = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_type") == fusion_type and p.get("fusion_method") == fusion_method and p.get("detected") and p.get("ground_truth"))
                fp = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_type") == fusion_type and p.get("fusion_method") == fusion_method and p.get("detected") and not p.get("ground_truth"))
                tn = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_type") == fusion_type and p.get("fusion_method") == fusion_method and not p.get("detected") and not p.get("ground_truth"))
                fn = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_type") == fusion_type and p.get("fusion_method") == fusion_method and not p.get("detected") and p.get("ground_truth"))
                acc, prec, rec, f1 = self.calculate_metrics(tp, tn, fp, fn)
                rmse_ft = self.compute_rmse_filtered(
                    method="fused",
                    fusion_type=fusion_type,
                    fusion_method=fusion_method,
                )
                print(f"  {fusion_type.upper():<4} {label:<25}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} RMSE={rmse_ft if rmse_ft is not None else 'N/A'}")

                for src in ["d1", "d2", "all"]:
                    tp = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_type") == fusion_type and p.get("fusion_method") == fusion_method and p.get("d_source") == src and p.get("detected") and p.get("ground_truth"))
                    fp = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_type") == fusion_type and p.get("fusion_method") == fusion_method and p.get("d_source") == src and p.get("detected") and not p.get("ground_truth"))
                    tn = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_type") == fusion_type and p.get("fusion_method") == fusion_method and p.get("d_source") == src and not p.get("detected") and not p.get("ground_truth"))
                    fn = sum(1 for p in self.rmse_pairs if p.get("method") == "fused" and p.get("fusion_type") == fusion_type and p.get("fusion_method") == fusion_method and p.get("d_source") == src and not p.get("detected") and p.get("ground_truth"))
                    acc, prec, rec, f1 = self.calculate_metrics(tp, tn, fp, fn)
                    rmse_src = self.compute_rmse_filtered(
                        method="fused",
                        fusion_type=fusion_type,
                        d_source=src,
                        fusion_method=fusion_method,
                    )
                    print(f"    {fusion_type.upper():<4} {label:<25} {src.upper():<3}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} RMSE={rmse_src if rmse_src is not None else 'N/A'}")
        print("\n" + "=" * 60)
