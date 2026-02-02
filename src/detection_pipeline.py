import cv2
import os
from pathlib import Path
from people_detector import PeopleDetector
from detection_statistics import DetectionStatistics

try:
    from config import *
except ImportError:
    # Default configuration if config.py is not found
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
    CROP_MIN_SIZE = 32


class DetectionPipeline:
    """Pipeline for batch processing images with people and weapon detection."""
    
    def __init__(self, model_path: str, person_confidence_threshold: float = 0.5, 
                 enable_weapon_detection: bool = True, weapon_confidence_threshold: float = 0.5, 
                 sample_majority_threshold: int = 1):
        """
        Initialize the detection pipeline.
        
        Args:
            model_path: Path to the YOLO model file
            person_confidence_threshold: Minimum confidence for person detections
            enable_weapon_detection: Whether to enable weapon detection on person crops
            weapon_confidence_threshold: Minimum confidence for weapon detections
            sample_majority_threshold: Number of frames with weapons needed to classify sample as having weapons
        """
        # Initialize the core detector
        self.detector = PeopleDetector(
            model_path=model_path,
            person_confidence_threshold=person_confidence_threshold,
            enable_weapon_detection=enable_weapon_detection,
            weapon_confidence_threshold=weapon_confidence_threshold
        )
        
        # Initialize statistics tracker with majority threshold
        self.stats = DetectionStatistics(sample_majority_threshold=sample_majority_threshold)
        
        # Pipeline settings
        self.save_crops = True  # Default to saving crops
        self.enable_weapon_detection = enable_weapon_detection
    
    def extract_person_crops(self, image, detections_info):
        crops = []
        
        for i, detection in enumerate(detections_info):
            x1, y1, x2, y2 = detection['bbox']
            person_confidence = detection.get('person_confidence', detection.get('confidence'))
            padding = 0.1
                
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
            
            # Crop the image
            cropped_person = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Store crop info
            crop_info = {
                'person_id': i + 1,
                'bbox': detection['bbox'],
                'padded_bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
                'person_confidence': person_confidence,
                'crop_size': (crop_width, crop_height)
            }
            crops.append((cropped_person, crop_info))
        
        return crops
    
    def detect_weapons_in_crops(self, crops_with_info):
        if not self.enable_weapon_detection or not self.detector.weapon_detector:
            return []
        
        return self.detector.weapon_detector.process_multiple_crops(crops_with_info)
    
    def draw_boxes_on_image(self, image, detections_info, weapon_results=None):
        result_image = image.copy()
        box_thickness = 2
        font_scale = 0.5
        font_thickness = 2
    
        # Draw person boxes in GREEN
        for detection in detections_info:
            x1, y1, x2, y2 = detection['bbox']
            person_confidence = detection.get('person_confidence', detection.get('confidence'))
            distance_m = detection.get('distance_m', None)
            dist_p = detection.get('distance_pinhole_m', None)
            dist_t = detection.get('distance_pitch_m', None)
            
            person_box_color = (0, 255, 0)  # Green for person
            cv2.rectangle(result_image, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        person_box_color, box_thickness)
            
            # Add confidence label
            label = f"Person: {person_confidence:.2f}"
            # Show both distance estimates when available for easy comparison.
            if dist_p is not None or dist_t is not None:
                if dist_p is not None:
                    label += f" P:{dist_p:.1f}m"
                if dist_t is not None:
                    label += f" Pitch:{dist_t:.1f}m"
            elif distance_m is not None:
                label += f" ({distance_m:.1f}m)"
            
            cv2.putText(result_image, label, 
                      (int(x1), int(y1) - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                      person_box_color, font_thickness)
        
        # Draw weapon boxes in RED (if any detected)
        if weapon_results:
            result_image = self.draw_weapon_boxes(result_image, weapon_results)
        
        return result_image

    def draw_weapon_boxes(self, image, weapon_results):
        box_thickness = 2
        font_scale = 0.5
        font_thickness = 2
            
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
                    full_wx2 = int(x2_pad + wx2)
                    full_wy2 = int(y2_pad + wy2)
                    
                    # Draw RED box for weapon
                    weapon_box_color = (0, 0, 255)  # Red for weapon
                    cv2.rectangle(image, 
                                (full_wx1, full_wy1), 
                                (full_wx2, full_wy2), 
                                weapon_box_color, box_thickness)
                    
                    # Add weapon label
                    weapon_label = f"{weapon_det['class']}: {weapon_det['confidence']:.2f}"
                    cv2.putText(image, weapon_label, 
                              (full_wx1, full_wy1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                              weapon_box_color, font_thickness)
        
        return image
    
    def save_bounding_box_crops(self, image, detections_info, crops_dir, base_filename):
        saved_crops = 0
        
        for i, detection in enumerate(detections_info):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection.get('confidence', detection.get('person_confidence'))
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
    
    def save_weapon_detection_results(self, weapon_results, output_dir, base_filename):
        if not weapon_results:
            return 0, 0
        
        # Create weapon detection output directory
        weapons_dir = os.path.join(output_dir, "weapon_detections")
        Path(weapons_dir).mkdir(parents=True, exist_ok=True)
        
        weapons_detected = 0
        people_with_weapons = 0
        
        for result in weapon_results:
            person_id = result['person_info']['person_id']
            person_confidence = result['person_info'].get('confidence', result['person_info'].get('person_confidence'))
            
            if result['has_weapons'] and result['weapon_crops']:
                people_with_weapons += 1
                
                # Save each weapon crop separately
                for weapon_idx, weapon_crop_info in enumerate(result['weapon_crops']):
                    weapon_crop = weapon_crop_info['crop']
                    weapon_confidence = weapon_crop_info.get('weapon_confidence', weapon_crop_info.get('confidence', 0.0))
                    weapon_class = weapon_crop_info['class']
                    
                    # Generate filename for weapon crop
                    weapon_filename = f"{base_filename}_person_{person_id:02d}_weapon_{weapon_idx+1:02d}_{weapon_class}_conf_{weapon_confidence:.2f}.jpg"
                    weapon_path = os.path.join(weapons_dir, weapon_filename)
                    
                    # Save the weapon crop
                    cv2.imwrite(weapon_path, weapon_crop)
                    weapons_detected += 1
        
        return weapons_detected, people_with_weapons
    
    def process_directory(self, input_dir: str, output_dir: str, with_weapons=False):
        # Create organized output directories
        detections_dir = os.path.join(output_dir, "detections")
        crops_dir = os.path.join(output_dir, "crops")
        
        Path(detections_dir).mkdir(parents=True, exist_ok=True)
        if self.save_crops:
            Path(crops_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine ground truth from filename convention when possible.
        dir_name = os.path.basename(input_dir)
        dir_meta = {}
        try:
            dir_meta = self.detector.extract_filename_metadata(input_dir)
        except Exception:
            dir_meta = {}
        sample_class_dir = (dir_meta or {}).get('sample_class')
        if sample_class_dir in ('real', 'falso'):
            has_weapons_ground_truth = (sample_class_dir == 'real')
        else:
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
                
                # Use detector to detect people
                _, detections = self.detector.detect_people(image_path, draw_boxes=False)

                # Extract testing-time condition metadata from filenames (used for RMSE comparisons).
                file_data = self.detector.extract_filename_metadata(image_path)
                real_distance = file_data.get('distance_m')
                camera_height_annotated = file_data.get('height_m')
                camera_pitch_annotated_deg = file_data.get('camera_pitch_deg')
                sample_class = file_data.get('sample_class') or ('real' if dir_name.lower().startswith('real') else 'falso')

                # No real camera pitch since we're not using telemetry
                camera_pitch_real_deg = None
                
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
                        weapon_results = self.detect_weapons_in_crops(person_crops)
                        weapons_detected, people_with_weapons_count = self.save_weapon_detection_results(weapon_results, output_dir, name)
                
                # Draw all boxes on the image
                combined_image = self.draw_boxes_on_image(original_image, detections, weapon_results)
                
                # Save result with all bounding boxes to detections folder
                output_filename = f"{name}_detected{ext}"
                detection_path = os.path.join(detections_dir, output_filename)
                
                cv2.imwrite(detection_path, combined_image)
                
                # Extract distances and (estimated, real) pairs from detections
                distances = [d['distance_m'] for d in detections if 'distance_m' in d]
                distance_pairs = []
                distance_pairs_pinhole = []
                distance_pairs_pitch = []
                if real_distance is not None:
                    for d in detections:
                        if 'distance_m' in d:
                            distance_pairs.append((d['distance_m'], real_distance))
                        if 'distance_pinhole_m' in d:
                            distance_pairs_pinhole.append((d['distance_pinhole_m'], real_distance))
                        if 'distance_pitch_m' in d:
                            distance_pairs_pitch.append((d['distance_pitch_m'], real_distance))
                # Update statistics with ground truth from directory name
                self.stats.add_image_results(
                    len(detections), weapons_detected, people_with_weapons_count, has_weapons_ground_truth,
                    distances, distance_pairs, real_distance, camera_height_annotated,
                    sample_class=sample_class,
                    camera_pitch_annotated_deg=camera_pitch_annotated_deg,
                    camera_pitch_real_deg=camera_pitch_real_deg,
                    distance_pairs_pinhole=distance_pairs_pinhole,
                    distance_pairs_pitch=distance_pairs_pitch,
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
    
    def process_all_sample_directories(self, samples_dir: str, output_base_dir: str, filter_clips=False):
        # Get all subdirectories in samples
        all_sample_dirs = [d for d in os.listdir(samples_dir) 
                          if os.path.isdir(os.path.join(samples_dir, d))]
        
        # Filter by clip numbers if requested
        if filter_clips:
            sample_dirs = [d for d in all_sample_dirs 
                          if any(f'_clip_00{i}' in d for i in [0, 2, 7])]
            print(f"Found {len(sample_dirs)} sample directories (filtered to clips 0,2,7 from {len(all_sample_dirs)} total)")
        else:
            sample_dirs = all_sample_dirs
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
            sample_meta = {}
            try:
                sample_meta = self.detector.extract_filename_metadata(sample_dir)
            except Exception:
                sample_meta = {}
            if (sample_meta or {}).get('sample_class') in ('real', 'falso'):
                sample_ground_truth = (sample_meta.get('sample_class') == 'real')
                sample_class = sample_meta.get('sample_class')
            else:
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
