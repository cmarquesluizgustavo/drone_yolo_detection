"""
Dual-Drone Detection Pipeline

Extends the single-drone pipeline to support synchronized processing
and fusion of detections from two drone perspectives.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Dict
from detection_pipeline import DetectionPipeline
from dual_drone_fusion import DualDroneFusion, FrameSynchronizer, Detection, DroneState


class DualDroneDetectionPipeline:
    """
    Pipeline for processing synchronized video streams from two drones.
    
    Implements:
    - Frame synchronization across drone streams
    - Parallel detection on both streams
    - Cross-drone detection association
    - Multi-view confidence fusion
    - Distance estimation fusion
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.4,
                 enable_weapon_detection: bool = True, 
                 weapon_confidence_threshold: float = 0.2,
                 sample_majority_threshold: int = 1,
                 association_threshold: float = 2.0):
        """
        Initialize dual-drone detection pipeline.
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for person detection
            enable_weapon_detection: Whether to detect weapons
            weapon_confidence_threshold: Minimum confidence for weapon detection
            sample_majority_threshold: Frames needed to classify sample as having weapons
            association_threshold: Distance threshold for cross-drone detection matching (meters)
        """
        # Create two independent detection pipelines (one per drone)
        self.pipeline_drone1 = DetectionPipeline(
            model_path, 
            confidence_threshold,
            enable_weapon_detection=enable_weapon_detection,
            weapon_confidence_threshold=weapon_confidence_threshold,
            sample_majority_threshold=sample_majority_threshold
        )
        
        self.pipeline_drone2 = DetectionPipeline(
            model_path,
            confidence_threshold,
            enable_weapon_detection=enable_weapon_detection,
            weapon_confidence_threshold=weapon_confidence_threshold,
            sample_majority_threshold=sample_majority_threshold
        )
        
        # Initialize fusion module
        self.fusion = DualDroneFusion(association_distance_threshold_m=association_threshold)
        
        # Initialize frame synchronizer
        self.synchronizer = FrameSynchronizer()
        
        # Share statistics between pipelines (use drone1's stats as primary)
        self.stats = self.pipeline_drone1.stats
        self.pipeline_drone2.stats = self.stats
        
        # Settings
        self.save_crops = True
        self.enable_weapon_detection = enable_weapon_detection
        
    def process_dual_drone_samples(self, input_dir_drone1: str, 
                                   input_dir_drone2: str,
                                   output_base_dir: str,
                                   filter_clips: bool = False):
        """
        Process all synchronized sample pairs from two drones.
        
        Args:
            input_dir_drone1: Base directory containing drone 1 samples
            input_dir_drone2: Base directory containing drone 2 samples
            output_base_dir: Base output directory
            filter_clips: If True, only process clips 0, 2, and 7 (0,45,-45)
        """
        # Get sample directories from both drones
        all_samples_drone1 = [d for d in os.listdir(input_dir_drone1)
                             if os.path.isdir(os.path.join(input_dir_drone1, d))]
        all_samples_drone2 = [d for d in os.listdir(input_dir_drone2)
                             if os.path.isdir(os.path.join(input_dir_drone2, d))]
        
        # Filter by clip numbers if requested
        if filter_clips:
            samples_drone1 = [d for d in all_samples_drone1
                            if any(f'_clip_00{i}' in d for i in [0, 2, 7])]
            samples_drone2 = [d for d in all_samples_drone2
                            if any(f'_clip_00{i}' in d for i in [0, 2, 7])]
        else:
            samples_drone1 = all_samples_drone1
            samples_drone2 = all_samples_drone2
        
        # Find matching sample pairs (same sample name in both directories)
        sample_pairs = []
        for sample1 in samples_drone1:
            if sample1 in samples_drone2:
                sample_pairs.append(sample1)
        
        print(f"Found {len(sample_pairs)} synchronized sample pairs")
        
        if len(sample_pairs) == 0:
            print("Warning: No matching sample pairs found between drone directories")
            return
        
        # Create output structure
        detections_base_dir = os.path.join(output_base_dir, "detections_dual_drone")
        crops_base_dir = os.path.join(output_base_dir, "crops_dual_drone")
        fused_base_dir = os.path.join(output_base_dir, "fused_detections")
        
        # Reset statistics for batch processing
        self.stats._in_batch_mode = True
        
        # Process each synchronized pair
        for sample_idx, sample_name in enumerate(sample_pairs, 1):
            print(f"\n Processing dual-drone sample {sample_idx}/{len(sample_pairs)}: {sample_name}")
            
            # Get paths
            sample_path_drone1 = os.path.join(input_dir_drone1, sample_name)
            sample_path_drone2 = os.path.join(input_dir_drone2, sample_name)
            
            # Determine ground truth from sample name
            sample_ground_truth = sample_name.lower().startswith("real")
            sample_class = 'real' if sample_name.lower().startswith('real') else 'falso'
            
            # Mark start of new sample
            self.stats.start_new_sample(sample_ground_truth, sample_class)
            
            # Process synchronized sample pair
            self.process_synchronized_sample_pair(
                sample_path_drone1,
                sample_path_drone2,
                output_base_dir,
                sample_name,
                detections_base_dir,
                crops_base_dir,
                fused_base_dir
            )
        
        # Finalize statistics
        self.stats.finalize()
        
    def process_synchronized_sample_pair(self, sample_dir_drone1: str,
                                        sample_dir_drone2: str,
                                        output_base: str,
                                        sample_name: str,
                                        detections_base: str,
                                        crops_base: str,
                                        fused_base: str):
        """
        Process a synchronized pair of sample directories from two drones.
        
        Args:
            sample_dir_drone1: Sample directory path for drone 1
            sample_dir_drone2: Sample directory path for drone 2
            output_base: Base output directory
            sample_name: Name of the sample
            detections_base: Base directory for detection outputs
            crops_base: Base directory for crop outputs
            fused_base: Base directory for fused outputs
        """
        # Get image files from both directories
        from config import SUPPORTED_FORMATS
        
        def get_image_files(directory):
            return sorted([
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.splitext(f)[1].lower() in [ext.lower() for ext in SUPPORTED_FORMATS]
            ])
        
        images_drone1 = get_image_files(sample_dir_drone1)
        images_drone2 = get_image_files(sample_dir_drone2)
        
        print(f"  Drone 1: {len(images_drone1)} frames")
        print(f"  Drone 2: {len(images_drone2)} frames")
        
        # Synchronize frames by filename/index
        synchronized_pairs = self.synchronizer.synchronize_by_frame_index(
            images_drone1, images_drone2
        )
        
        print(f"  Synchronized: {len(synchronized_pairs)} frame pairs")
        
        # Create output directories for this sample
        sample_det_d1 = os.path.join(detections_base, f"{sample_name}_drone1")
        sample_det_d2 = os.path.join(detections_base, f"{sample_name}_drone2")
        sample_det_fused = os.path.join(fused_base, sample_name)
        sample_crops_d1 = os.path.join(crops_base, f"{sample_name}_drone1")
        sample_crops_d2 = os.path.join(crops_base, f"{sample_name}_drone2")
        
        for dir_path in [sample_det_d1, sample_det_d2, sample_det_fused,
                        sample_crops_d1, sample_crops_d2]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Process each synchronized frame pair
        for frame_idx, (frame1_path, frame2_path) in enumerate(synchronized_pairs):
            self._process_frame_pair(
                frame1_path, frame2_path, frame_idx,
                sample_det_d1, sample_det_d2, sample_det_fused,
                sample_crops_d1, sample_crops_d2,
                sample_name
            )
    
    def _process_frame_pair(self, frame1_path: str, frame2_path: str, frame_idx: int,
                           output_det_d1: str, output_det_d2: str, output_fused: str,
                           output_crops_d1: str, output_crops_d2: str,
                           sample_name: str):
        """
        Process a single synchronized frame pair from two drones.
        
        Args:
            frame1_path: Path to frame from drone 1
            frame2_path: Path to frame from drone 2
            frame_idx: Frame index
            output_det_d1: Output directory for drone 1 detections
            output_det_d2: Output directory for drone 2 detections
            output_fused: Output directory for fused detections
            output_crops_d1: Output directory for drone 1 crops
            output_crops_d2: Output directory for drone 2 crops
            sample_name: Name of the sample
        """
        # Load images
        img1 = cv2.imread(frame1_path)
        img2 = cv2.imread(frame2_path)
        
        # Get filenames
        filename1 = os.path.basename(frame1_path)
        filename2 = os.path.basename(frame2_path)
        name1, ext1 = os.path.splitext(filename1)
        name2, ext2 = os.path.splitext(filename2)
        
        # Detect people in both frames
        _, detections1 = self.pipeline_drone1.detector.detect_people(frame1_path, draw_boxes=False)
        _, detections2 = self.pipeline_drone2.detector.detect_people(frame2_path, draw_boxes=False)
        
        # Process crops and weapon detection for both drones
        weapon_results1 = []
        weapon_results2 = []
        
        if detections1:
            crops1 = self.pipeline_drone1.extract_person_crops(img1, detections1)
            if self.save_crops:
                self.pipeline_drone1.save_bounding_box_crops(img1, detections1, output_crops_d1, name1)
            if self.enable_weapon_detection:
                weapon_results1 = self.pipeline_drone1.detect_weapons_in_crops(crops1)
        
        if detections2:
            crops2 = self.pipeline_drone2.extract_person_crops(img2, detections2)
            if self.save_crops:
                self.pipeline_drone2.save_bounding_box_crops(img2, detections2, output_crops_d2, name2)
            if self.enable_weapon_detection:
                weapon_results2 = self.pipeline_drone2.detect_weapons_in_crops(crops2)
        
        # Draw detections on individual frames
        img1_annotated = self.pipeline_drone1.draw_boxes_on_image(img1, detections1, weapon_results1)
        img2_annotated = self.pipeline_drone2.draw_boxes_on_image(img2, detections2, weapon_results2)
        
        # Save individual drone detections
        cv2.imwrite(os.path.join(output_det_d1, f"{name1}_detected{ext1}"), img1_annotated)
        cv2.imwrite(os.path.join(output_det_d2, f"{name2}_detected{ext2}"), img2_annotated)
        
        # Fuse detections across drones
        fused_detections = self._fuse_frame_detections(
            detections1, detections2, weapon_results1, weapon_results2, frame_idx
        )
        
        # Create visualization of fused detections (side-by-side view)
        fused_vis = self._create_fused_visualization(
            img1_annotated, img2_annotated, fused_detections
        )
        
        # Save fused visualization
        fused_filename = f"{sample_name}_frame_{frame_idx:04d}_fused.jpg"
        cv2.imwrite(os.path.join(output_fused, fused_filename), fused_vis)
        
        # Update statistics
        has_weapons_gt = sample_name.lower().startswith("real")
        weapons_detected_d1 = sum(1 for r in weapon_results1 if r.get('has_weapons', False))
        weapons_detected_d2 = sum(1 for r in weapon_results2 if r.get('has_weapons', False))
        
        # Count fused weapon detections
        weapons_fused = sum(1 for d in fused_detections if d.get('has_weapon', False))
        
        # Update stats for both individual detections and fused
        sample_class = 'real' if has_weapons_gt else 'falso'
        
        # Individual drone stats
        self.stats.add_image_results(
            len(detections1), weapons_detected_d1, 
            len([r for r in weapon_results1 if r.get('has_weapons', False)]),
            has_weapons_gt, [], [], None, None, sample_class, None
        )
        
        self.stats.add_image_results(
            len(detections2), weapons_detected_d2,
            len([r for r in weapon_results2 if r.get('has_weapons', False)]),
            has_weapons_gt, [], [], None, None, sample_class, None
        )
        
        # Fused detection stats
        self.stats.add_image_results(
            len(fused_detections), weapons_fused,
            weapons_fused,
            has_weapons_gt, [], [], None, None, sample_class, None
        )
        
        # Print progress
        print(f"    Frame {frame_idx:04d}: D1={len(detections1)}p/{weapons_detected_d1}w, "
              f"D2={len(detections2)}p/{weapons_detected_d2}w, "
              f"Fused={len(fused_detections)}p/{weapons_fused}w")
    
    def _fuse_frame_detections(self, detections1: List[Dict], detections2: List[Dict],
                              weapon_results1: List[Dict], weapon_results2: List[Dict],
                              frame_idx: int) -> List[Dict]:
        """
        Fuse detections from both drones for a single frame.
        
        Args:
            detections1: Person detections from drone 1
            detections2: Person detections from drone 2
            weapon_results1: Weapon detection results from drone 1
            weapon_results2: Weapon detection results from drone 2
            frame_idx: Frame index
            
        Returns:
            List of fused detection dictionaries
        """
        # Convert to Detection objects for fusion
        # For now, use simplified ground-plane coordinates (to be extended with actual GPS)
        detection_objs1 = []
        for i, det in enumerate(detections1):
            has_weapon = False
            weapon_conf = 0.0
            if i < len(weapon_results1):
                has_weapon = weapon_results1[i].get('has_weapons', False)
                if has_weapon and weapon_results1[i].get('weapon_detections'):
                    weapon_conf = max([w['confidence'] for w in weapon_results1[i]['weapon_detections']])
            
            # Simplified: use image center as (0,0) and bbox position as relative coords
            distance = det.get('distance_m', 10.0)  # Default if not available
            bearing = 0.0  # Simplified
            
            detection_objs1.append(Detection(
                bbox=tuple(det['bbox']),
                confidence=det['confidence'],
                distance_m=distance,
                bearing_deg=bearing,
                ground_x=distance * np.sin(np.radians(bearing)),
                ground_y=distance * np.cos(np.radians(bearing)),
                gps_lat=0.0,  # Placeholder
                gps_lon=0.0,  # Placeholder
                drone_id=1,
                frame_id=frame_idx,
                has_weapon=has_weapon,
                weapon_confidence=weapon_conf
            ))
        
        detection_objs2 = []
        for i, det in enumerate(detections2):
            has_weapon = False
            weapon_conf = 0.0
            if i < len(weapon_results2):
                has_weapon = weapon_results2[i].get('has_weapons', False)
                if has_weapon and weapon_results2[i].get('weapon_detections'):
                    weapon_conf = max([w['confidence'] for w in weapon_results2[i]['weapon_detections']])
            
            distance = det.get('distance_m', 10.0)
            bearing = 0.0
            
            detection_objs2.append(Detection(
                bbox=tuple(det['bbox']),
                confidence=det['confidence'],
                distance_m=distance,
                bearing_deg=bearing,
                ground_x=distance * np.sin(np.radians(bearing)) + 10.0,  # Offset for drone 2
                ground_y=distance * np.cos(np.radians(bearing)),
                gps_lat=0.0,
                gps_lon=0.0,
                drone_id=2,
                frame_id=frame_idx,
                has_weapon=has_weapon,
                weapon_confidence=weapon_conf
            ))
        
        # Perform fusion
        fused = self.fusion.associate_detections(detection_objs1, detection_objs2)
        
        return fused
    
    def _create_fused_visualization(self, img1: np.ndarray, img2: np.ndarray,
                                   fused_detections: List[Dict]) -> np.ndarray:
        """
        Create a side-by-side visualization with fusion information.
        
        Args:
            img1: Annotated image from drone 1
            img2: Annotated image from drone 2
            fused_detections: List of fused detections
            
        Returns:
            Combined visualization image
        """
        # Create side-by-side layout
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Make heights equal
        max_h = max(h1, h2)
        if h1 < max_h:
            img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if h2 < max_h:
            img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Concatenate horizontally
        combined = np.hstack([img1, img2])
        
        # Add fusion information text overlay
        info_y = 30
        for i, det in enumerate(fused_detections):
            source = det.get('source', 'unknown')
            conf = det.get('confidence', 0.0)
            has_weapon = det.get('has_weapon', False)
            weapon_conf = det.get('weapon_confidence', 0.0)
            
            info_text = f"Det {i+1}: {source} | conf={conf:.2f}"
            if has_weapon:
                info_text += f" | WEAPON={weapon_conf:.2f}"
            
            cv2.putText(combined, info_text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
        
        # Add title
        cv2.putText(combined, "Drone 1", (w1//2 - 50, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(combined, "Drone 2", (w1 + w2//2 - 50, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        return combined
