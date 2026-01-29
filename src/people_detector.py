"""
YOLOv11 People Detector
Core detection functionality for detecting people in images.
"""

import cv2
import os
import logging
from ultralytics import YOLO
from distance_estimation import Camera

try:
    from config import *
except ImportError:
    # Default configuration if config.py is not found
    PERSON_CLASS_ID = 0
    BOX_COLOR = (0, 255, 0)
    BOX_THICKNESS = 2
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    CROP_PADDING = 0.1
    CROP_MIN_SIZE = 32

try:
    from weapon_detector import WeaponDetector
    WEAPON_DETECTION_AVAILABLE = True
except ImportError as e:
    WEAPON_DETECTION_AVAILABLE = False
    print(f"Warning: Weapon detection not available. Error: {e}")


class PeopleDetector:
    """Core detector class for person and weapon detection in single images."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.4, 
                 enable_weapon_detection: bool = True, weapon_confidence_threshold: float = 0.2):
        """
        Initialize the people detector with YOLOv11 model.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence for detections
            enable_weapon_detection: Whether to enable weapon detection on person crops
            weapon_confidence_threshold: Minimum confidence for weapon detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = PERSON_CLASS_ID
        
        # Initialize weapon detector if available and enabled
        self.weapon_detector = None
        self.enable_weapon_detection = enable_weapon_detection and WEAPON_DETECTION_AVAILABLE
        self.weapon_confidence_threshold = weapon_confidence_threshold
        
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
        
        # Initialize camera for distance estimation
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
    
    def extract_real_distance_from_filename(self, filepath: str):
        """Extract the real distance from the filename/folder structure."""
        try:
            dir_path = os.path.dirname(filepath)
            dir_name = os.path.basename(dir_path)
            
            if not dir_name:
                dir_name = os.path.splitext(os.path.basename(filepath))[0]
            parts = dir_name.split('_')

            for i in range(len(parts) - 1):
                if parts[i + 1].isdigit() and i + 2 < len(parts) and parts[i + 2].isdigit():
                    distance = float(parts[i + 1])
                    return distance
            
            return None
        except (ValueError, IndexError):
            return None
        
    def extract_camera_height_from_filename(self, filepath: str):
        """
        Extracts the camera height from folder name.
        Pattern: class_distance_height_clip_...
        Example: falso_05_02_clip_000 -> distance=05, height=02
        """
        import re
        dir_path = os.path.dirname(filepath)
        dir_name = os.path.basename(dir_path)
        if not dir_name:
            dir_name = os.path.splitext(os.path.basename(filepath))[0]
        numbers = re.findall(r'\d{2,3}', dir_name)
        # numbers[0] = distance (05), numbers[1] = height (02), numbers[2] = clip number (000)
        if len(numbers) >= 2:
            try:
                height = float(numbers[1])
                return height
            except ValueError:
                pass
        return None
    
    def extract_camera_tilt_from_filename(self, filepath: str):
        """
        Extracts the camera tilt from folder name.
        Pattern: class_distance_height_TILT_clip_...
        Example: falso_05_02_2_clip_000 -> distance=05, height=02, tilt=2
        If no tilt is present, returns None.
        """
        import re
        dir_path = os.path.dirname(filepath)
        dir_name = os.path.basename(dir_path)
        if not dir_name:
            dir_name = os.path.splitext(os.path.basename(filepath))[0]
        numbers = re.findall(r'\d{1,3}', dir_name)
        # If we have 4+ numbers: numbers[0]=distance, numbers[1]=height, numbers[2]=tilt, numbers[3]=clip
        # If we have 3 numbers: numbers[0]=distance, numbers[1]=height, numbers[2]=clip (no tilt)
        if len(numbers) >= 4:
            try:
                tilt = float(numbers[2])
                return tilt
            except ValueError:
                pass
        return None
        
    def detect_people(self, image_path: str, draw_boxes: bool = False):
        """
        Detect people in a single image.
        
        Args:
            image_path: Path to the input image
            draw_boxes: Whether to draw boxes on the image (for combined visualization)
            
        Returns:
            tuple: (image_with_boxes, detections_info)
        """
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
                        
                        distance_m = None
                        if self.camera:
                            try:
                                # Extract real distance, camera height, and tilt from file path
                                image_name = os.path.basename(image_path)
                                real_distance_m = self.extract_real_distance_from_filename(image_path)
                                camera_height_m = self.extract_camera_height_from_filename(image_path)
                                camera_tilt_deg = self.extract_camera_tilt_from_filename(image_path)
                                
                                # Use estimate_distance_2 if tilt is available, otherwise use pinhole model
                                if camera_tilt_deg is not None and camera_height_m is not None:
                                    y_bottom = y2
                                    distance_m = self.camera.estimate_distance_2(
                                        y_pixel=y_bottom,
                                        camera_tilt_deg=camera_tilt_deg,
                                        camera_height_m=camera_height_m
                                    )
                                else:
                                    distance_m = self.camera.estimate_distance(person_height_px)
                                
                                # Build log message with proper formatting
                                real_dist_str = f"{real_distance_m:.2f}" if real_distance_m is not None else "N/A"
                                cam_height_str = f"{camera_height_m:.2f}" if camera_height_m is not None else "N/A"
                                cam_tilt_str = f"{camera_tilt_deg:.2f}" if camera_tilt_deg is not None else "N/A"
                                log_message = (f"Image: {image_name}, Person: {person_idx + 1}, "
                                             f"PixelHeight: {person_height_px:.1f}px, "
                                             f"Estimated: {distance_m:.2f}m, "
                                             f"Real: {real_dist_str}m, "
                                             f"CameraHeight: {cam_height_str}m, "
                                             f"CameraTilt: {cam_tilt_str}deg, "
                                             f"Confidence: {confidence:.3f}, "
                                             f"BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                                self.distance_logger.info(log_message)
                                
                                # Enhanced console output
                                console_msg = f"  -> Person {person_idx + 1}: Est:{distance_m:.2f}m"
                                if real_distance_m is not None:
                                    console_msg += f", Real:{real_distance_m:.2f}m"
                                if camera_height_m is not None:
                                    console_msg += f", CamHeight:{camera_height_m:.2f}m"
                                if camera_tilt_deg is not None:
                                    console_msg += f", Tilt:{camera_tilt_deg:.0f}deg"
                                console_msg += f", {person_height_px:.1f}px"
                                print(console_msg)
                                
                            except Exception as e:
                                print(f"Warning: Failed to estimate distance for person {person_idx + 1}: {e}")
                        
                        # Draw bounding box in GREEN for person (only if draw_boxes is True)
                        if draw_boxes:
                            person_box_color = (0, 255, 0)  # Green for person
                            cv2.rectangle(image_with_boxes, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        person_box_color, BOX_THICKNESS)
                            
                            # Add confidence label
                            label = f"Person: {confidence:.2f}"
                            if distance_m is not None:
                                label += f" ({distance_m:.1f}m)"
                            
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
                        if distance_m is not None:
                            detection_info['distance_m'] = float(distance_m)
                        
                        detections_info.append(detection_info)
        
        return image_with_boxes, detections_info
