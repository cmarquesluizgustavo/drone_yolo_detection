class DetectionStatistics:
    """Class to track comprehensive detection statistics."""
    
    def __init__(self, sample_majority_threshold=1):
        """
        Initialize statistics tracker.
        
        Args:
            sample_majority_threshold: Number of frames with weapon detections 
                                      needed to classify a sample as having weapons
        """
        self.sample_majority_threshold = sample_majority_threshold
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_images = 0
        self.images_with_people = 0
        self.total_people = 0
        self.total_weapons = 0
        self.people_with_weapons = 0
        self.total_samples = 0
        self.samples_with_weapons = 0
        self.current_sample_has_weapons = False
        
        # Per-frame metrics
        self.tp_frame = 0
        self.tn_frame = 0
        self.fp_frame = 0
        self.fn_frame = 0
        
        # Per-sample metrics
        self.tp_sample = 0
        self.tn_sample = 0
        self.fp_sample = 0
        self.fn_sample = 0
        
        # Per-sample metrics by class
        self.sample_metrics_by_class = {}  # {'real': {tp, tn, fp, fn}, 'falso': {...}}
        
        # Legacy single metrics (will be same as frame metrics)
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0
        self.prec= 0
        self.recall = 0
        self.f1score = 0
        
        # Distance tracking
        self.distances = []
        self.people_with_distance = 0
        # For RMSE: list of (estimated, real) pairs
        self.distance_pairs = []

        # Method-specific RMSE evaluation
        self.distance_pairs_pinhole = []  # (est_pinhole, real)
        self.distance_pairs_pitch = []     # (est_pitch_based, real)

        # Method-specific RMSE evaluation by (class, distance, height, camera_pitch)
        self.distance_pairs_pinhole_by_combo_pitch = {}  # {(cls, dist, height, pitch): [(est, real), ...]}
        self.distance_pairs_pitch_by_combo_pitch = {}     # {(cls, dist, height, pitch): [(est, real), ...]}

        # Distance pairs per camera height for RMSE calculation
        self.distance_pairs_by_height = {}  # {2: [(est, real), ...], 5: [...]}
        # Distance pairs by (class, distance, height) combinations
        self.distance_pairs_by_combination = {}  # {('real', 5, 2): [(est, real), ...], ...}
        # Distance pairs by (distance, height) for aggregation across classes
        self.distance_pairs_by_dist_height = {}  # {(5, 2): [(est, real), ...], ...}
        # Distance pairs by (class, distance, height, camera_pitch) combinations
        self.distance_pairs_by_combination_with_pitch = {}  # {('real', 5, 2, 45): [(est, real), ...], ...}
        # Distance pairs by (distance, height, camera_pitch) for aggregation across classes
        self.distance_pairs_by_dist_height_pitch = {}  # {(5, 2, 45): [(est, real), ...], ...}
        
        # Segmented metrics: per distance, height, and class
        self.metrics_by_distance = {}  # {5: {tp, tn, fp, fn}, 10: {...}}
        self.metrics_by_height = {}    # {2: {tp, tn, fp, fn}, 5: {...}}
        self.metrics_by_class = {}     # {'real': {tp, tn, fp, fn}, 'falso': {...}}
        
        # Sample tracking
        self.current_sample_ground_truth = False
        self.current_sample_detected_weapons = False
        self.current_sample_class = None
        self.current_sample_frames_with_weapons = 0
        self.current_sample_total_frames = 0
    
    def start_new_sample(self, sample_ground_truth=False, sample_class=None):
        """Mark the start of a new sample directory."""
        # Finalize previous sample if this isn't the first one
        if hasattr(self, 'current_sample_ground_truth'):
            self.finalize_current_sample()
        
        # Initialize new sample
        self.current_sample_ground_truth = sample_ground_truth
        self.current_sample_detected_weapons = False
        self.current_sample_has_weapons = False
        self.current_sample_class = sample_class
        self.current_sample_frames_with_weapons = 0
        self.current_sample_total_frames = 0
        self.total_samples += 1
    
    def finalize_current_sample(self):
        """Finalize the current sample and update sample-level metrics using majority voting."""
        if not hasattr(self, 'current_sample_ground_truth'):
            return
        
        # Determine if sample has weapons using majority voting
        # Sample is classified as having weapons if >= threshold frames have weapons
        sample_has_weapons = self.current_sample_frames_with_weapons >= self.sample_majority_threshold
        
        # Update sample count
        if sample_has_weapons:
            self.samples_with_weapons += 1
        
        # Update sample-level confusion matrix
        if sample_has_weapons:  # Weapons detected in sample (via majority voting)
            if self.current_sample_ground_truth:  # Ground truth: should have weapons
                metric_result = 'tp'
                self.tp_sample += 1  # True Positive
            else:  # Ground truth: should not have weapons
                metric_result = 'fp'
                self.fp_sample += 1  # False Positive
        else:  # No weapons detected in sample
            if self.current_sample_ground_truth:  # Ground truth: should have weapons
                metric_result = 'fn'
                self.fn_sample += 1  # False Negative
            else:  # Ground truth: should not have weapons
                metric_result = 'tn'
                self.tn_sample += 1  # True Negative
        
        # Update per-sample metrics by class
        if self.current_sample_class is not None:
            if self.current_sample_class not in self.sample_metrics_by_class:
                self.sample_metrics_by_class[self.current_sample_class] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            self.sample_metrics_by_class[self.current_sample_class][metric_result] += 1
    
    def add_image_results(self, num_people, num_weapons, people_with_weapons_count, has_weapons_ground_truth,
                          distances=None, distance_pairs=None, real_distance=None, camera_height=None,
                          sample_class=None,
                          camera_pitch_annotated_deg=None,
                          camera_pitch_real_deg=None,
                          distance_pairs_pinhole=None,
                          distance_pairs_pitch=None):
        """Add results from processing one image."""

        self.total_images += 1
        if num_people > 0:
            self.images_with_people += 1
        
        self.total_people += num_people
        self.total_weapons += num_weapons
        self.people_with_weapons += people_with_weapons_count
        
        # Track sample-level detection (for majority voting)
        self.current_sample_total_frames += 1
        if num_weapons > 0:
            self.current_sample_has_weapons = True
            self.current_sample_detected_weapons = True
            self.current_sample_frames_with_weapons += 1
        
        # Calculate per-frame metrics
        detected_weapon = num_weapons > 0
        
        if detected_weapon:  # Weapons detected in frame
            if has_weapons_ground_truth:  # Ground truth: should have weapons
                metric_result = 'tp'
                self.tp_frame += 1  # True Positive
            else:  # Ground truth: should not have weapons
                metric_result = 'fp'
                self.fp_frame += 1  # False Positive
        else:  # No weapons detected in frame
            if has_weapons_ground_truth:  # Ground truth: should have weapons
                metric_result = 'fn'
                self.fn_frame += 1  # False Negative
            else:  # Ground truth: should not have weapons
                metric_result = 'tn'
                self.tn_frame += 1  # True Negative
        
        # Update segmented metrics
        if real_distance is not None:
            if real_distance not in self.metrics_by_distance:
                self.metrics_by_distance[real_distance] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            self.metrics_by_distance[real_distance][metric_result] += 1
        
        if camera_height is not None:
            if camera_height not in self.metrics_by_height:
                self.metrics_by_height[camera_height] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            self.metrics_by_height[camera_height][metric_result] += 1
        
        if sample_class is not None:
            if sample_class not in self.metrics_by_class:
                self.metrics_by_class[sample_class] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            self.metrics_by_class[sample_class][metric_result] += 1
        
        # Update legacy metrics (same as frame metrics for backward compatibility)
        self.tp = self.tp_frame
        self.tn = self.tn_frame
        self.fp = self.fp_frame
        self.fn = self.fn_frame
        
        # Add distance information
        if distances:
            self.distances.extend(distances)
            self.people_with_distance += len(distances)
        # Add (estimated, real) pairs for RMSE
        if distance_pairs:
            self.distance_pairs.extend(distance_pairs)
            # Also track by camera height for RMSE
            if camera_height is not None:
                if camera_height not in self.distance_pairs_by_height:
                    self.distance_pairs_by_height[camera_height] = []
                self.distance_pairs_by_height[camera_height].extend(distance_pairs)
            # Track by (class, distance, height) combination
            if sample_class is not None and real_distance is not None and camera_height is not None:
                combination_key = (sample_class, real_distance, camera_height)
                if combination_key not in self.distance_pairs_by_combination:
                    self.distance_pairs_by_combination[combination_key] = []
                self.distance_pairs_by_combination[combination_key].extend(distance_pairs)
            # Track by (distance, height) for aggregation across classes
            if real_distance is not None and camera_height is not None:
                dist_height_key = (real_distance, camera_height)
                if dist_height_key not in self.distance_pairs_by_dist_height:
                    self.distance_pairs_by_dist_height[dist_height_key] = []
                self.distance_pairs_by_dist_height[dist_height_key].extend(distance_pairs)
            # Track by (class, distance, height, camera pitch) combination
            if sample_class is not None and real_distance is not None and camera_height is not None and camera_pitch_annotated_deg is not None:
                combination_key_pitch = (sample_class, real_distance, camera_height, camera_pitch_annotated_deg)
                if combination_key_pitch not in self.distance_pairs_by_combination_with_pitch:
                    self.distance_pairs_by_combination_with_pitch[combination_key_pitch] = []
                self.distance_pairs_by_combination_with_pitch[combination_key_pitch].extend(distance_pairs)
            # Track by (distance, height, camera pitch) for aggregation across classes
            if real_distance is not None and camera_height is not None and camera_pitch_annotated_deg is not None:
                dist_height_pitch_key = (real_distance, camera_height, camera_pitch_annotated_deg)
                if dist_height_pitch_key not in self.distance_pairs_by_dist_height_pitch:
                    self.distance_pairs_by_dist_height_pitch[dist_height_pitch_key] = []
                self.distance_pairs_by_dist_height_pitch[dist_height_pitch_key].extend(distance_pairs)

        # Store method-specific pairs (overall only; use these to compare methods).
        if distance_pairs_pinhole:
            self.distance_pairs_pinhole.extend(distance_pairs_pinhole)
        if distance_pairs_pitch:
            self.distance_pairs_pitch.extend(distance_pairs_pitch)

        # Store method-specific pairs by (class, distance, height, camera pitch) when possible.
        if sample_class is not None and real_distance is not None and camera_height is not None and camera_pitch_annotated_deg is not None:
            combo_key = (sample_class, real_distance, camera_height, camera_pitch_annotated_deg)

            if distance_pairs_pinhole:
                if combo_key not in self.distance_pairs_pinhole_by_combo_pitch:
                    self.distance_pairs_pinhole_by_combo_pitch[combo_key] = []
                self.distance_pairs_pinhole_by_combo_pitch[combo_key].extend(distance_pairs_pinhole)

            if distance_pairs_pitch:
                if combo_key not in self.distance_pairs_pitch_by_combo_pitch:
                    self.distance_pairs_pitch_by_combo_pitch[combo_key] = []
                self.distance_pairs_pitch_by_combo_pitch[combo_key].extend(distance_pairs_pitch)
    def compute_rmse(self, distance_pairs=None):
        """Compute RMSE for distance estimation (only where real distance is available)."""
        pairs = distance_pairs if distance_pairs is not None else self.distance_pairs
        if not pairs:
            return None
        diffsq = [(est-real)**2 for est, real in pairs if real is not None]
        if not diffsq:
            return None
        mse = sum(diffsq) / len(diffsq)
        return mse ** 0.5
    
    def finalize(self):
        """Finalize statistics (call at the end)."""
        # Finalize the last sample
        self.finalize_current_sample()
    
    def calculate_metrics(self, tp, tn, fp, fn):
        """Calculate accuracy, precision, recall, and F1-score from confusion matrix."""
        total_predictions = tp + tn + fp + fn
        
        if total_predictions > 0:
            accuracy = (tp + tn) / total_predictions
        else:
            accuracy = 0
            
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if (precision + recall) > 0:
            f1score = 2 * (precision * recall) / (precision + recall)
        else:
            f1score = 0
        
        return accuracy, precision, recall, f1score
    
    def get_percentages(self):
        """Calculate and return percentage statistics."""
        # Percentage of images with people
        people_in_images_pct = (self.images_with_people / self.total_images * 100) if self.total_images > 0 else 0
        
        # Percentage of people with weapons
        weapons_in_people_pct = (self.people_with_weapons / self.total_people * 100) if self.total_people > 0 else 0
        
        # Percentage of samples with weapons
        weapons_in_samples_pct = (self.samples_with_weapons / self.total_samples * 100) if self.total_samples > 0 else 0
        
        # Calculate frame-level metrics
        frame_accuracy, frame_precision, frame_recall, frame_f1score = self.calculate_metrics(
            self.tp_frame, self.tn_frame, self.fp_frame, self.fn_frame
        )
        
        # Calculate sample-level metrics
        sample_accuracy, sample_precision, sample_recall, sample_f1score = self.calculate_metrics(
            self.tp_sample, self.tn_sample, self.fp_sample, self.fn_sample
        )
        
        # Update legacy metrics (frame-level for backward compatibility)
        self.accuracy = frame_accuracy
        self.precision = frame_precision
        self.recall = frame_recall
        self.f1score = frame_f1score
        
        return {
            'people_in_images_pct': people_in_images_pct,
            'weapons_in_people_pct': weapons_in_people_pct,
            'weapons_in_samples_pct': weapons_in_samples_pct,
            'total_images': self.total_images,
            'images_with_people': self.images_with_people,
            'total_people': self.total_people,
            'total_weapons': self.total_weapons,
            'people_with_weapons': self.people_with_weapons,
            'total_samples': self.total_samples,
            'samples_with_weapons': self.samples_with_weapons,
            
            # Frame-level metrics
            'frame_accuracy': frame_accuracy,
            'frame_precision': frame_precision,
            'frame_recall': frame_recall,
            'frame_f1score': frame_f1score,
            'tp_frame': self.tp_frame,
            'tn_frame': self.tn_frame,
            'fp_frame': self.fp_frame,
            'fn_frame': self.fn_frame,
            
            # Sample-level metrics
            'sample_accuracy': sample_accuracy,
            'sample_precision': sample_precision,
            'sample_recall': sample_recall,
            'sample_f1score': sample_f1score,
            'tp_sample': self.tp_sample,
            'tn_sample': self.tn_sample,
            'fp_sample': self.fp_sample,
            'fn_sample': self.fn_sample,
            
            # Legacy metrics (same as frame)
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1score': self.f1score,
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            
            'people_with_distance': self.people_with_distance,
            'total_distances': len(self.distances)
        }
    
    def print_summary(self):
        """Print comprehensive statistics summary."""
        stats = self.get_percentages()
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DETECTION STATISTICS")
        print("=" * 60)
        
        print(f"IMAGE STATISTICS:")
        print(f"   Total images processed: {stats['total_images']:,}")
        print(f"   Images with people: {stats['images_with_people']:,} ({stats['people_in_images_pct']:.1f}%)")
        
        print(f"PEOPLE STATISTICS:")
        print(f"   Total people detected: {stats['total_people']:,}")
        print(f"   People with weapons: {stats['people_with_weapons']:,} ({stats['weapons_in_people_pct']:.1f}%)")
        
        print(f"WEAPON STATISTICS:")
        print(f"   Total weapons detected: {stats['total_weapons']:,}")
        
        print(f"SAMPLE STATISTICS:")
        print(f"   Total samples processed: {stats['total_samples']:,}")
        print(f"   Samples with weapons: {stats['samples_with_weapons']:,} ({stats['weapons_in_samples_pct']:.1f}%)")
        
        if stats['total_distances'] > 0:
            print(f"   People with distance data: {stats['people_with_distance']:,}")
        else:
            print(f"   No distance data available")
        
        print(f"KEY PERCENTAGES:")
        print(f"People in images: {stats['people_in_images_pct']:.1f}% of images contain people")
        print(f"Weapons in people: {stats['weapons_in_people_pct']:.1f}% of people have weapons")
        print(f"Weapons in samples: {stats['weapons_in_samples_pct']:.1f}% of samples contain weapons")
        
        print(f"\nOVERALL METRICS:")
        print(f"   Accuracy:  {stats['frame_accuracy']:.3f}")
        print(f"   Precision: {stats['frame_precision']:.3f}")
        print(f"   Recall:    {stats['frame_recall']:.3f}")
        print(f"   F1-Score:  {stats['frame_f1score']:.3f}")
        print(f"   TP: {stats['tp_frame']}, TN: {stats['tn_frame']}, FP: {stats['fp_frame']}, FN: {stats['fn_frame']}")
        
        # Print segmented metrics
        if self.metrics_by_distance:
            print(f"\nMETRICS BY DISTANCE:")
            for dist in sorted(self.metrics_by_distance.keys()):
                m = self.metrics_by_distance[dist]
                acc, prec, rec, f1 = self.calculate_metrics(m['tp'], m['tn'], m['fp'], m['fn'])
                print(f"   Distance: {dist}m")
                print(f"      Accuracy:  {acc:.3f}")
                print(f"      Precision: {prec:.3f}")
                print(f"      Recall:    {rec:.3f}")
                print(f"      F1-Score:  {f1:.3f}")
                print(f"      TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")
        
        if self.metrics_by_height:
            print(f"\nMETRICS BY CAMERA HEIGHT:")
            for height in sorted(self.metrics_by_height.keys()):
                m = self.metrics_by_height[height]
                acc, prec, rec, f1 = self.calculate_metrics(m['tp'], m['tn'], m['fp'], m['fn'])
                # Calculate RMSE for this height
                rmse_height = None
                if height in self.distance_pairs_by_height:
                    rmse_height = self.compute_rmse(self.distance_pairs_by_height[height])
                print(f"   Height: {height}m")
                print(f"      Accuracy:  {acc:.3f}")
                print(f"      Precision: {prec:.3f}")
                print(f"      Recall:    {rec:.3f}")
                print(f"      F1-Score:  {f1:.3f}")
                if rmse_height is not None:
                    print(f"      RMSE:      {rmse_height:.3f}m")
                print(f"      TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")
        
        if self.metrics_by_class:
            print(f"\nMETRICS BY CLASS:")
            for cls in sorted(self.metrics_by_class.keys()):
                m = self.metrics_by_class[cls]
                acc, prec, rec, f1 = self.calculate_metrics(m['tp'], m['tn'], m['fp'], m['fn'])
                print(f"   Class: {cls}")
                print(f"      Accuracy:  {acc:.3f}")
                print(f"      Precision: {prec:.3f}")
                print(f"      Recall:    {rec:.3f}")
                print(f"      F1-Score:  {f1:.3f}")
                print(f"      TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")
        
        # Print RMSE by (distance, height) combinations, showing per-class and aggregate
        if self.distance_pairs_by_dist_height:
            print(f"\nRMSE BY (DISTANCE, HEIGHT) COMBINATIONS:")
            for (dist, height) in sorted(self.distance_pairs_by_dist_height.keys()):
                print(f"   Distance: {dist}m, Height: {height}m")
                # Show per-class RMSE
                for cls in ['falso', 'real']:
                    cls_key = (cls, dist, height)
                    if cls_key in self.distance_pairs_by_combination:
                        pairs = self.distance_pairs_by_combination[cls_key]
                        rmse_cls = self.compute_rmse(pairs)
                        if rmse_cls is not None:
                            print(f"      Class '{cls}': RMSE = {rmse_cls:.3f}m ({len(pairs)} measurements)")
                # Show aggregate RMSE for all classes
                all_pairs = self.distance_pairs_by_dist_height[(dist, height)]
                rmse_all = self.compute_rmse(all_pairs)
                if rmse_all is not None:
                    print(f"      Class 'all':   RMSE = {rmse_all:.3f}m ({len(all_pairs)} measurements)")
        
        # Print RMSE by (distance, height, camera pitch) combinations, showing per-class and aggregate
        if self.distance_pairs_by_dist_height_pitch:
            print(f"\nRMSE BY (DISTANCE, HEIGHT, CAMERA PITCH) COMBINATIONS:")
            for (dist, height, pitch) in sorted(self.distance_pairs_by_dist_height_pitch.keys()):
                print(f"   Distance: {dist}m, Height: {height}m, CameraPitch: {pitch}deg")
                # Show per-class RMSE
                for cls in ['falso', 'real']:
                    cls_key = (cls, dist, height, pitch)
                    if cls_key in self.distance_pairs_by_combination_with_pitch:
                        pairs = self.distance_pairs_by_combination_with_pitch[cls_key]
                        rmse_cls = self.compute_rmse(pairs)
                        if rmse_cls is not None:
                            print(f"      Class '{cls}': RMSE = {rmse_cls:.3f}m ({len(pairs)} measurements)")
                # Show aggregate RMSE for all classes
                all_pairs = self.distance_pairs_by_dist_height_pitch[(dist, height, pitch)]
                rmse_all = self.compute_rmse(all_pairs)
                if rmse_all is not None:
                    print(f"      Class 'all':   RMSE = {rmse_all:.3f}m ({len(all_pairs)} measurements)")
        
        # Print overall RMSE
        rmse = self.compute_rmse()
        if rmse is not None:
            print(f"\nOVERALL DISTANCE ESTIMATION RMSE: {rmse:.3f}m")

        # Print method comparison RMSE (pinhole vs pitch-based)
        rmse_pinhole = self.compute_rmse(self.distance_pairs_pinhole)
        rmse_pitch = self.compute_rmse(self.distance_pairs_pitch)
        if rmse_pinhole is not None or rmse_pitch is not None:
            print("\nDISTANCE ESTIMATION METHOD COMPARISON:")
            if rmse_pinhole is not None:
                print(f"   PINHOLE RMSE: {rmse_pinhole:.3f}m ({len(self.distance_pairs_pinhole)} measurements)")
            else:
                print("   PINHOLE RMSE: N/A")
            if rmse_pitch is not None:
                print(f"   PITCH-BASED RMSE: {rmse_pitch:.3f}m ({len(self.distance_pairs_pitch)} measurements)")
            else:
                print("   PITCH-BASED RMSE: N/A")

        # Method comparison by (class, distance, height, camera pitch)
        if self.distance_pairs_pinhole_by_combo_pitch or self.distance_pairs_pitch_by_combo_pitch:
            print("\nDISTANCE METHOD RMSE BY (CLASS, DISTANCE, HEIGHT, CAMERA PITCH):")
            all_keys = sorted(set(self.distance_pairs_pinhole_by_combo_pitch.keys()) | set(self.distance_pairs_pitch_by_combo_pitch.keys()))
            for (cls, dist, height, pitch) in all_keys:
                pairs_p = self.distance_pairs_pinhole_by_combo_pitch.get((cls, dist, height, pitch), [])
                pairs_pitch = self.distance_pairs_pitch_by_combo_pitch.get((cls, dist, height, pitch), [])
                rmse_p = self.compute_rmse(pairs_p)
                rmse_pitch = self.compute_rmse(pairs_pitch)

                # Skip buckets with no usable measurements
                if rmse_p is None and rmse_pitch is None:
                    continue

                p_str = f"{rmse_p:.3f}m" if rmse_p is not None else "N/A"
                pitch_str = f"{rmse_pitch:.3f}m" if rmse_pitch is not None else "N/A"
                print(f"   {cls} | dist={dist}m, h={height}m, camera_pitch={pitch}deg -> PINHOLE={p_str}, PITCH-BASED={pitch_str}")
        
        print("\n" + "=" * 60)
        
        print(f"\nPER-SAMPLE METRICS (Majority Threshold: {self.sample_majority_threshold} frame(s)):")
        print(f"   Accuracy:  {stats['sample_accuracy']:.3f}")
        print(f"   Precision: {stats['sample_precision']:.3f}")
        print(f"   Recall:    {stats['sample_recall']:.3f}")
        print(f"   F1-Score:  {stats['sample_f1score']:.3f}")
        print(f"   TP: {stats['tp_sample']}, TN: {stats['tn_sample']}, FP: {stats['fp_sample']}, FN: {stats['fn_sample']}")
        
        # Print per-sample metrics by class
        if self.sample_metrics_by_class:
            print(f"\nPER-SAMPLE METRICS BY CLASS (Majority Threshold: {self.sample_majority_threshold} frame(s)):")
            for cls in sorted(self.sample_metrics_by_class.keys()):
                m = self.sample_metrics_by_class[cls]
                acc, prec, rec, f1 = self.calculate_metrics(m['tp'], m['tn'], m['fp'], m['fn'])
                print(f"   Class: {cls}")
                print(f"      Accuracy:  {acc:.3f}")
                print(f"      Precision: {prec:.3f}")
                print(f"      Recall:    {rec:.3f}")
                print(f"      F1-Score:  {f1:.3f}")
                print(f"      TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")
        
        print("\n" + "=" * 60)
        print(f"   F1-Score: {stats['sample_f1score']:.4f}")
        print(f"   TP: {stats['tp_sample']}")
        print(f"   TN: {stats['tn_sample']}")
        print(f"   FP: {stats['fp_sample']}")
        print(f"   FN: {stats['fn_sample']}")
        

        '''
        if stats['total_weapons'] > 0:
            avg_weapons_per_person = stats['total_weapons'] / stats['people_with_weapons']
            print(f"\n   📊 Average weapons per armed person: {avg_weapons_per_person:.1f}")
        '''

        # Print RMSE for distance estimation
        rmse = self.compute_rmse()
        if rmse is not None:
            print(f"\nDistance Estimation RMSE: {rmse:.3f} meters")
        else:
            print(f"\nDistance Estimation RMSE: N/A (no ground truth)")

        print("\n" + "=" * 60)
