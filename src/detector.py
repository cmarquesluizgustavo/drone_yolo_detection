"""
YOLO People Detection Pipeline with Weapon Detection
Processes images and detects people, then detects weapons in person crops.
"""

import re
import cv2
import os
import logging
from pathlib import Path
from ultralytics import YOLO
from estimation import Camera

try:
    from config import *
except ImportError:
    # Default configuration if config.py is not found
    PERSON_CLASS_ID = 0
    BOX_COLOR = (0, 255, 0)
    BOX_THICKNESS = 2
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

try:
    from weapon_detector import WeaponDetector
    WEAPON_DETECTION_AVAILABLE = True
except ImportError as e:
    WEAPON_DETECTION_AVAILABLE = False
    print(f"Warning: Weapon detection not available. Error: {e}")

class DetectionStatistics:
    """Class to track comprehensive detection statistics."""

    def __init__(self, sample_majority_threshold=1):
        """
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
        
        # Distance tracking
        self.distances = []
        self.people_with_distance = 0
        
        # For RMSE: list of (estimated, real) pairs
        self.distance_pairs_pinhole = []
        self.distance_pairs_tilt = []

        # Distance pairs per camera height for RMSE calculation
        self.distance_pairs_by_height = {}  # {2: [(est, real), ...], 5: [...]}
        
        # Track tilt angles used for each camera height
        self.tilt_angles_by_height = {}  # {2: [45.0, 45.0, ...], 5: [...]}
        
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
    
    def add_image_results(self, num_people, num_weapons, people_with_weapons_count, has_weapons_ground_truth, distances=None, distance_pairs=None, real_distance=None, camera_height=None, sample_class=None, camera_tilt_deg=None):
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
        
        # Add distance information
        if distances:
            self.distances.extend(distances)
            self.people_with_distance += len(distances)
        
        if distance_pairs:
            for est, real, method in distance_pairs:
                if method == "pinhole":
                    self.distance_pairs_pinhole.append((est, real))
                elif method == "tilt":
                    self.distance_pairs_tilt.append((est, real))
                    
                    # Track tilt angle used for tilt method
                    if camera_height is not None and camera_tilt_deg is not None:
                        if camera_height not in self.tilt_angles_by_height:
                            self.tilt_angles_by_height[camera_height] = []
                        self.tilt_angles_by_height[camera_height].append(camera_tilt_deg)

                # group by camera height
                if camera_height is not None:
                    if camera_height not in self.distance_pairs_by_height:
                        self.distance_pairs_by_height[camera_height] = {"pinhole": [], "tilt": []}

                    self.distance_pairs_by_height[camera_height][method].append((est, real))

                
    def compute_rmse(self, distance_pairs):
        if not distance_pairs:
            return None

        diffsq = [(est - real) ** 2 for est, real in distance_pairs if real is not None]

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
            
            'people_with_distance': self.people_with_distance,
            'total_distances': len(self.distances)
        }
    
    def print_summary(self):
        """Print comprehensive statistics summary."""
        stats = self.get_percentages()
        
        print("\\n" + "=" * 60)
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
                print(f"   Height: {height}m")
                print(f"      Accuracy:  {acc:.3f}")
                print(f"      Precision: {prec:.3f}")
                print(f"      Recall:    {rec:.3f}")
                print(f"      F1-Score:  {f1:.3f}")
                print(f"      TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")

                for method in ["pinhole", "tilt"]:
                    pairs = self.distance_pairs_by_height.get(height, {}).get(method, [])
                    rmse = self.compute_rmse(pairs)

                    if rmse is not None:
                        print(f"   {method.upper()} RMSE: {rmse:.3f} m")
                        
                        # For tilt method, show the tilt angle(s) used
                        if method == "tilt" and height in self.tilt_angles_by_height:
                            tilt_angles = self.tilt_angles_by_height[height]
                            if tilt_angles:
                                # Calculate mean and show range
                                mean_tilt = sum(tilt_angles) / len(tilt_angles)
                                min_tilt = min(tilt_angles)
                                max_tilt = max(tilt_angles)
                                if min_tilt == max_tilt:
                                    print(f"   TILT ANGLE USED: {mean_tilt:.1f}°")
                                else:
                                    print(f"   TILT ANGLE USED: {mean_tilt:.1f}° (range: {min_tilt:.1f}° - {max_tilt:.1f}°)")
                    else:
                        print(f"   {method.upper()} RMSE: N/A")
        
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
        
        rmse_pinhole = self.compute_rmse(self.distance_pairs_pinhole)
        rmse_tilt = self.compute_rmse(self.distance_pairs_tilt)

        print("\nGLOBAL DISTANCE RMSE:")
        if rmse_pinhole is not None:
            print(f"   Pinhole Method RMSE: {rmse_pinhole:.3f} m")
        if rmse_tilt is not None:
            print(f"   Tilt Method RMSE: {rmse_tilt:.3f} m")
        
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

    def find_best_tilt_by_height_and_distance(self, camera, height, detections, real_distance):
        best_rmse = float("inf")
        best_tilt = None

        for tilt in range(0, 90):
            pairs = []

            for det in detections:
                if "bbox" not in det:
                    continue

                x1, y1, x2, y2 = det["bbox"]
                y_bottom = y2

                est = camera.estimate_distance_2(
                    y_pixel=y_bottom,
                    camera_tilt_deg=tilt,
                    camera_height_m=height
                )

                if est is not None and real_distance is not None:
                    pairs.append((est, real_distance))

            rmse = self.compute_rmse(pairs)

            if rmse is not None and rmse < best_rmse:
                best_rmse = rmse
                best_tilt = tilt

        return best_tilt




class PeopleDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.4, enable_weapon_detection: bool = True, weapon_confidence_threshold: float = 0.2, sample_majority_threshold: int = 1):
        """
        Initialize the people detector with YOLOv11 model.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence for detections
            enable_weapon_detection: Whether to enable weapon detection on person crops
            weapon_confidence_threshold: Minimum confidence for weapon detections
            sample_majority_threshold: Number of frames with weapons needed to classify sample as having weapons
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        # COCO class ID for 'person' is 0
        self.person_class_id = PERSON_CLASS_ID
        self.save_crops = True  # Default to saving crops
        
        # Initialize statistics tracker with majority threshold
        self.stats = DetectionStatistics(sample_majority_threshold=sample_majority_threshold)
        
        # Initialize weapon detector if available and enabled
        self.weapon_detector = None
        self.enable_weapon_detection = enable_weapon_detection and WEAPON_DETECTION_AVAILABLE
        self.weapon_confidence_threshold = weapon_confidence_threshold

        self.last_tilt = None
        self.last_height = None
        self.last_distance = None


        
        if self.enable_weapon_detection:
            try:
                self.weapon_detector = WeaponDetector(confidence_threshold=weapon_confidence_threshold)
                print(f"Weapon detection enabled (confidence: {weapon_confidence_threshold})")
            except Exception as e:
                print(f"Failed to initialize weapon detector: {e}")
                self.enable_weapon_detection = False
        else:
            if enable_weapon_detection and not WEAPON_DETECTION_AVAILABLE:
                print("Weapon detection requested but not available")
            else:
                print("Weapon detection disabled")
        self.camera = None
        
        try:
            self.camera = Camera()  # Using default drone camera settings
            self.distance_logger = logging.getLogger('distance_logger')
            self.distance_logger.setLevel(logging.INFO)
            
            if not self.distance_logger.handlers:
                log_file = 'person_distances.log'
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.distance_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Failed to initialize distance estimation: {e}")


    def extract_distance_height_tilt(self, filepath):
        """file name format => class_distance_height_tilt_clip_number"""

        distance = height = tilt = None

        try:
            dir_path = os.path.dirname(filepath)
            dir_name = os.path.basename(dir_path)

            if not dir_name:
                dir_name = os.path.splitext(os.path.basename(filepath))[0]

            parts = dir_name.split('_')

            # Find numeric or "None" tokens
            values = []
            for p in parts:
                if p == "None":
                    values.append(None)
                elif re.fullmatch(r'\d+(\.\d+)?', p):
                    values.append(float(p))

            # Assign values safely
            if len(values) > 0:
                distance = values[0]
            if len(values) > 1:
                height = values[1]
            if len(values) > 2:
                tilt = values[2]

        except (ValueError, IndexError):
            pass

        return distance, height, tilt


        
    def detect_people(self, image_path: str, draw_boxes: bool = False):
        """Detect people in a single image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Run inference - only detect person class (class 0)
        results = self.model(image, imgsz=640, iou=0.6, conf=self.confidence_threshold, classes=[0], verbose=False)
        
        # Process results
        detections_info = []
        image_with_boxes = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for person_idx, box in enumerate(boxes):
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only process person detections above threshold
                    if class_id == self.person_class_id and confidence >= self.confidence_threshold:
                        # height pixels
                        person_height_px = y2 - y1
                        distance_pinhole = self.camera.estimate_distance(person_height_px)
                        
                        # Extract real distance and camera height from file path
                        real_distance_m, camera_height_m, camera_tilt_deg = self.extract_distance_height_tilt(image_path)

                        y_bottom = y2  # bottom of bounding box

                        distance_tilt = None

                        if camera_height_m is not None and real_distance_m is not None:

                            # If tilt is given in filename → trust it
                            if camera_tilt_deg is not None:
                                distance_tilt = self.camera.estimate_distance_2(
                                    y_pixel=y_bottom,
                                    camera_tilt_deg=camera_tilt_deg,
                                    camera_height_m=camera_height_m
                                )

                            else:
                                # Recalculate tilt if height OR distance changed
                                need_recalc = (
                                    self.last_tilt is None or
                                    self.last_height != camera_height_m or
                                    self.last_distance != real_distance_m
                                )

                                if need_recalc:
                                    best_tilt = self.stats.find_best_tilt_by_height_and_distance(
                                        camera=self.camera,
                                        height=camera_height_m,
                                        detections=[{
                                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                                        }],
                                        real_distance=real_distance_m
                                    )

                                    if best_tilt is not None:
                                        self.last_tilt = best_tilt
                                        self.last_height = camera_height_m
                                        self.last_distance = real_distance_m

                                        print(
                                            f"[AUTO-TILT] Recalculated tilt: {best_tilt}° "
                                            f"(Height={camera_height_m}, Distance={real_distance_m})"
                                        )

                                # Use stored tilt
                                if self.last_tilt is not None:
                                    distance_tilt = self.camera.estimate_distance_2(
                                        y_pixel=y_bottom,
                                        camera_tilt_deg=self.last_tilt,
                                        camera_height_m=camera_height_m
                                    )

                        
                        # Console print
                        tilt_str = f"{distance_tilt:.2f}m" if distance_tilt is not None else "N/A"
                        print(f"  -> Person {person_idx + 1}: Pinhole:{distance_pinhole:.2f}m, Tilt:{tilt_str}, Real:{real_distance_m:.2f}m")

                                
                        # Draw bounding box in GREEN for person (only if draw_boxes is True)
                        if draw_boxes:
                            person_box_color = (0, 255, 0)  # Green for person
                            cv2.rectangle(image_with_boxes, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        person_box_color, BOX_THICKNESS)
                            
                            # Add confidence label
                            label = f"Person: {confidence:.2f}"
                            if distance_pinhole is not None:
                                label += f", distance_pinhole: {distance_pinhole:.2f}m"

                            if distance_tilt is not None:
                                label += f", distance_tilt: {distance_tilt:.2f}m"

                            
                            cv2.putText(image_with_boxes, label, 
                                      (int(x1), int(y1) - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 
                                      person_box_color, FONT_THICKNESS)
                        
                        # Store detection info
                        detection_info = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': 'person',
                            'height_px': float(person_height_px)
                        }
                        
                        # Add distance if available
                        if distance_pinhole is not None:
                            detection_info['distance_pinhole_m'] = float(distance_pinhole)

                        if distance_tilt is not None:
                            detection_info['distance_tilt_m'] = float(distance_tilt)

                        detections_info.append(detection_info)
        
        return image_with_boxes, detections_info
    
    def extract_person_crops(self, image, detections_info):
        """Extract person crops from the original image based on detections."""
        crops = []
        
        for i, detection in enumerate(detections_info):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Add padding to the bounding box
            try:
                padding = CROP_PADDING
            except NameError:
                padding = 0.1  # Default fallback
                
            width = x2 - x1
            height = y2 - y1
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            # Calculate padded coordinates
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(image.shape[1], x2 + pad_x)
            y2_pad = min(image.shape[0], y2 + pad_y)
            
            crop_width = x2_pad - x1_pad
            crop_height = y2_pad - y1_pad
            
            # Crop the image
            cropped_person = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Store crop info
            crop_info = {
                'person_id': i + 1,
                'bbox': detection['bbox'],
                'padded_bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
                'confidence': confidence,
                'crop_size': (crop_width, crop_height)
            }
            crops.append((cropped_person, crop_info))
        
        return crops
    
    def detect_weapons_in_crops(self, crops_with_info):
        """Detect weapons in person crops using the weapon detector."""
        if not self.enable_weapon_detection or not self.weapon_detector:
            return []
        
        return self.weapon_detector.process_multiple_crops(crops_with_info)
    
    def draw_weapon_boxes_on_full_image(self, image, weapon_results):
        """Draw weapon bounding boxes in RED on the full image."""
        for result in weapon_results:
            if result['has_weapons'] and result['weapon_detections']:
                # Get person crop info to translate weapon boxes to full image coordinates
                person_info = result['person_info']
                padded_bbox = person_info['padded_bbox']
                x1_pad, y1_pad, x2_pad, y2_pad = padded_bbox
                
                # Draw each weapon detection
                for weapon_det in result['weapon_detections']:
                    # Weapon bbox is relative to the person crop
                    wx1, wy1, wx2, wy2 = weapon_det['bbox']
                    
                    # Translate to full image coordinates
                    full_wx1 = int(x1_pad + wx1)
                    full_wy1 = int(y1_pad + wy1)
                    full_wx2 = int(x1_pad + wx2)
                    full_wy2 = int(y1_pad + wy2)
                    
                    # Draw RED box for weapon
                    weapon_box_color = (0, 0, 255)  # Red for weapon
                    cv2.rectangle(image, 
                                (full_wx1, full_wy1), 
                                (full_wx2, full_wy2), 
                                weapon_box_color, BOX_THICKNESS)
                    
                    # Add weapon label
                    weapon_label = f"{weapon_det['class']}: {weapon_det['confidence']:.2f}"
                    cv2.putText(image, weapon_label, 
                              (full_wx1, full_wy1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 
                              weapon_box_color, FONT_THICKNESS)
        
        return image
    
    def save_weapon_detection_results(self, weapon_results, output_dir, base_filename):
        """Save weapon detection results - only weapon bounding box crops."""
        if not weapon_results:
            return 0, 0
        
        # Create weapon detection output directory
        weapons_dir = os.path.join(output_dir, "weapon_detections")
        Path(weapons_dir).mkdir(parents=True, exist_ok=True)
        
        weapons_detected = 0
        people_with_weapons = 0
        
        for result in weapon_results:
            person_id = result['person_info']['person_id']
            #person_confidence = result['person_info']['confidence']
            
            if result['has_weapons'] and result['weapon_crops']:
                people_with_weapons += 1
                
                # Save each weapon crop separately
                for weapon_idx, weapon_crop_info in enumerate(result['weapon_crops']):
                    weapon_crop = weapon_crop_info['crop']
                    weapon_confidence = weapon_crop_info['confidence']
                    weapon_class = weapon_crop_info['class']
                    
                    # Generate filename for weapon crop
                    weapon_filename = f"{base_filename}_person_{person_id:02d}_weapon_{weapon_idx+1:02d}_{weapon_class}_conf_{weapon_confidence:.2f}.jpg"
                    weapon_path = os.path.join(weapons_dir, weapon_filename)
                    
                    # Save the weapon crop
                    cv2.imwrite(weapon_path, weapon_crop)
                    weapons_detected += 1
        
        return weapons_detected, people_with_weapons

    def save_bounding_box_crops(self, image, detections_info, crops_dir, base_filename):
        """Save individual cropped images for each detected person."""
        
        saved_crops = 0
        
        for i, detection in enumerate(detections_info):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Add padding to the bounding box
            try:
                padding = CROP_PADDING
            except NameError:
                padding = 0.1  # Default fallback
                
            width = x2 - x1
            height = y2 - y1
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            # Calculate padded coordinates
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(image.shape[1], x2 + pad_x)
            y2_pad = min(image.shape[0], y2 + pad_y)
            
            # Check minimum size
            crop_width = x2_pad - x1_pad
            crop_height = y2_pad - y1_pad
            
            try:
                min_size = CROP_MIN_SIZE
            except NameError:
                min_size = 32  # Default fallback
                
            if crop_width < min_size or crop_height < min_size:
                continue  # Skip crops that are too small
            
            # Crop the image
            cropped_person = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Generate filename for the crop
            crop_filename = f"{base_filename}_person_{i+1:02d}_conf_{confidence:.2f}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            
            # Save the cropped image
            cv2.imwrite(crop_path, cropped_person)
            saved_crops += 1
        
        return saved_crops
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory and save results."""
        # Create organized output directories
        detections_dir = os.path.join(output_dir, "detections")
        crops_dir = os.path.join(output_dir, "crops")
        
        Path(detections_dir).mkdir(parents=True, exist_ok=True)
        if self.save_crops:
            Path(crops_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine ground truth from directory name
        dir_name = os.path.basename(input_dir)
        has_weapons_ground_truth = dir_name.lower().startswith("real")
        
        # Get all image files
        image_files = []
        for file in os.listdir(input_dir):
            if os.path.splitext(file)[1].lower() in [ext.lower() for ext in SUPPORTED_FORMATS]:
                image_files.append(os.path.join(input_dir, file))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        print(f"Ground truth for this directory: {'Has weapons' if has_weapons_ground_truth else 'No weapons'}")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            try:
                #print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # Load original image for cropping
                original_image = cv2.imread(image_path)
                
                # Get filename parts early for use in saving
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                
                # Detect people (without drawing boxes yet - we'll draw everything together)
                image_with_boxes, detections = self.detect_people(image_path, draw_boxes=False)
                
                # Extract person crops for weapon detection
                person_crops = []
                crops_saved = 0
                weapons_detected = 0
                people_with_weapons_count = 0
                weapon_results = []
                
                if detections:
                    # Extract person crops
                    person_crops = self.extract_person_crops(original_image, detections)
                    
                    # Save individual bounding box crops to crops folder
                    if self.save_crops:
                        crops_saved = self.save_bounding_box_crops(original_image, detections, crops_dir, name)
                    
                    # Perform weapon detection on person crops
                    if self.enable_weapon_detection and person_crops:
                        #print(f"  -> Checking {len(person_crops)} person crops for weapons...")
                        weapon_results = self.detect_weapons_in_crops(person_crops)
                        weapons_detected, people_with_weapons_count = self.save_weapon_detection_results(weapon_results, output_dir, name)
                
                # Now draw ALL boxes on the same image: GREEN for persons, RED for weapons
                combined_image = original_image.copy()
                
                # Draw person boxes in GREEN
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    #distance_m = detection.get('distance_m', None)
                    
                    person_box_color = (0, 255, 0)  # Green for person
                    cv2.rectangle(combined_image, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                person_box_color, BOX_THICKNESS)
                    
                    # Add confidence label
                    label = f"Person: {confidence:.2f}"

                    if 'distance_pixel_m' in detection:
                        label += f" Pinhole:{detection['distance_pixel_m']:.1f}m"

                    if 'distance_tilt_m' in detection:
                        label += f" Tilt:{detection['distance_tilt_m']:.1f}m"

                    
                    cv2.putText(combined_image, label, 
                              (int(x1), int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 
                              person_box_color, FONT_THICKNESS)
                
                # Draw weapon boxes in RED (if any detected)
                if weapon_results:
                    combined_image = self.draw_weapon_boxes_on_full_image(combined_image, weapon_results)
                
                # Save result with all bounding boxes to detections folder
                output_filename = f"{name}_detected{ext}"
                detection_path = os.path.join(detections_dir, output_filename)
                
                cv2.imwrite(detection_path, combined_image)
                
                # Extract distances and (estimated, real) pairs from detections
                distances = []
                for d in detections:
                    if 'distance_pinhole_m' in d:
                        distances.append(d['distance_pinhole_m'])

                    if 'distance_tilt_m' in d and d['distance_tilt_m'] is not None:
                        distances.append(d['distance_tilt_m'])
                
                real_distance, camera_height, camera_tilt = self.extract_distance_height_tilt(image_path)
                sample_class = 'real' if dir_name.lower().startswith('real') else 'falso'
                
                distance_pairs = []
                if real_distance is not None:
                    for d in detections:
                        if 'distance_pinhole_m' in d:
                            distance_pairs.append((d['distance_pinhole_m'], real_distance, "pinhole"))

                        if 'distance_tilt_m' in d and d['distance_tilt_m'] is not None:
                            distance_pairs.append((d['distance_tilt_m'], real_distance, "tilt"))

                # Update statistics with ground truth from directory name
                self.stats.add_image_results(
                    len(detections),
                    weapons_detected,
                    people_with_weapons_count,
                    has_weapons_ground_truth,
                    distances,
                    distance_pairs,
                    real_distance,
                    camera_height,
                    sample_class,
                    camera_tilt,
                )

                
                # Print detection summary
                if detections:
                    summary_parts = [f"Found {len(detections)} people"]
                    if self.save_crops:
                        summary_parts.append(f"saved {crops_saved} crops")
                    if self.enable_weapon_detection:
                        summary_parts.append(f"detected {weapons_detected} weapons")
                        if people_with_weapons_count > 0:
                            summary_parts.append(f"({people_with_weapons_count} people with weapons)")
                    ground_truth_label = "real weapons" if has_weapons_ground_truth else "no weapons"
                    summary_parts.append(f"ground: {ground_truth_label}")
                    print(f"  -> {', '.join(summary_parts)}")
                else:
                    ground_truth_label = "real weapons" if has_weapons_ground_truth else "no weapons"
                    print(f"  -> No people detected, ground: {ground_truth_label}")

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Print directory summary for single directory processing
        if not hasattr(self.stats, '_in_batch_mode'):
            print(f"\\nDirectory processing complete!")
            self.stats.print_summary()
            stats = self.stats.get_percentages()
            print(f"Images processed: {stats['total_images']}")
            print(f"People detected: {stats['total_people']}")
            if self.enable_weapon_detection:
                print(f"Weapons detected: {stats['total_weapons']}")
                if stats['total_people'] > 0:
                    print(f"Weapon rate: {stats['weapons_in_people_pct']:.1f}% of people have weapons")
            if stats['total_distances'] > 0:
                print(f"Distance measurements: {stats['total_distances']}")
    
    def process_all_sample_directories(self, samples_dir: str, output_base_dir: str):
        """Process all sample directories and maintain organized folder structure."""
        # Get all subdirectories in samples
        sample_dirs = [d for d in os.listdir(samples_dir) 
                      if os.path.isdir(os.path.join(samples_dir, d))]
        
        print(f"Found {len(sample_dirs)} sample directories")
        
        # Create main organized output structure
        detections_base_dir = os.path.join(output_base_dir, "detections")
        crops_base_dir = os.path.join(output_base_dir, "crops")
        
        # Reset statistics for batch processing
        #self.stats.reset()
        self.stats._in_batch_mode = True  # Flag to indicate batch processing
        
        for sample_idx, sample_dir in enumerate(sample_dirs, 1):
            input_path = os.path.join(samples_dir, sample_dir)
            
            # Create organized output paths for this sample directory
            sample_detections_dir = os.path.join(detections_base_dir, sample_dir)
            sample_crops_dir = os.path.join(crops_base_dir, sample_dir)
            
            # Create temporary output structure for this sample
            temp_output = os.path.join(output_base_dir, "temp", sample_dir)
            
            print(f"\n Processing sample {sample_idx}/{len(sample_dirs)}: {sample_dir}")
            
            # Determine ground truth and class for this sample
            sample_ground_truth = sample_dir.lower().startswith("real")
            sample_class = 'real' if sample_dir.lower().startswith('real') else 'falso'
            
            # Mark start of new sample for statistics with ground truth and class
            self.stats.start_new_sample(sample_ground_truth, sample_class)
            
            # Process directory - ground truth is now determined from filenames
            self.process_directory(input_path, temp_output)
            
            # Move results to organized structure
            self._organize_sample_output(temp_output, sample_detections_dir, sample_crops_dir)
        
        # Finalize statistics
        self.stats.finalize()
        
        # Clean up empty weapon detection directories
        self._cleanup_empty_weapon_directories(output_base_dir)
   
    
        


    def _organize_sample_output(self, temp_dir: str, detections_dir: str, crops_dir: str):
        import os    
        import shutil
        from pathlib import Path
        import stat
        
        """
        Move processed files from temp directory to organized structure.
        Handles Windows permission issues.
        """

        def on_rm_error(func, path, exc_info):
            """Force delete read-only files (Windows quirk)."""
            os.chmod(path, stat.S_IWRITE)
            func(path)

        temp_detections = os.path.join(temp_dir, "detections")
        temp_crops = os.path.join(temp_dir, "crops")
        temp_weapons = os.path.join(temp_dir, "weapon_detections")

        # Move detections
        if os.path.exists(temp_detections):
            Path(detections_dir).mkdir(parents=True, exist_ok=True)
            for file in os.listdir(temp_detections):
                src = os.path.join(temp_detections, file)
                dst = os.path.join(detections_dir, file)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Move crops
        if os.path.exists(temp_crops) and getattr(self, "save_crops", True):
            Path(crops_dir).mkdir(parents=True, exist_ok=True)
            for file in os.listdir(temp_crops):
                src = os.path.join(temp_crops, file)
                dst = os.path.join(crops_dir, file)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Move weapon detections
        if os.path.exists(temp_weapons):
            weapons_base_dir = os.path.dirname(crops_dir).replace("\\crops", "\\weapon_detections").replace("/crops", "/weapon_detections")
            weapons_dir = os.path.join(weapons_base_dir, os.path.basename(crops_dir))
            Path(weapons_dir).mkdir(parents=True, exist_ok=True)
            for file in os.listdir(temp_weapons):
                src = os.path.join(temp_weapons, file)
                dst = os.path.join(weapons_dir, file)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Clean up temp directory (safe version)
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, onerror=on_rm_error)
            except PermissionError:
                print(f"[WARNING] Could not remove {temp_dir} due to permission issues.")

    
    def _cleanup_empty_weapon_directories(self, output_base_dir: str):
        """
        Remove empty weapon detection directories after processing is complete.
        """
        weapons_base_dir = os.path.join(output_base_dir, "weapon_detections")
        
        if not os.path.exists(weapons_base_dir):
            return
        
        # Find and remove empty directories
        empty_dirs = []
        for root, dirs, files in os.walk(weapons_base_dir, topdown=False):
            # Skip the base weapon_detections directory itself
            if root == weapons_base_dir:
                continue
                
            # Check if directory is empty (no files)
            if not files:
                empty_dirs.append(root)
        
        # Remove empty directories
        removed_count = 0
        for empty_dir in empty_dirs:
            try:
                os.rmdir(empty_dir)
                removed_count += 1
            except OSError:
                # Directory might not be empty or have permission issues
                pass
        
        if removed_count > 0:
            print(f"\nCleaned up {removed_count} empty weapon detection directories")
