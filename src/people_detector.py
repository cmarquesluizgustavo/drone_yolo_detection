"""
YOLOv11 People Detector
Core detection functionality for detecting people in images.
"""

import cv2
import os
import logging
import re
from pathlib import PurePath
from ultralytics import YOLO
from camera import Camera

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
    logging.getLogger(__name__).warning("Weapon detection not available: %s", e)


class PeopleDetector:
    """Core detector class for person and weapon detection in single images."""

    _FILENAME_META_RE = re.compile(
        r"(?i)(real|falso)_(\d{1,3}(?:\.\d+)?)_(\d{1,3}(?:\.\d+)?)_(\d{1,3}(?:\.\d+)?)(?=(_|\.|$))"
    )
    
    def __init__(self, model_path: str, person_confidence_threshold: float = 0.4, 
                 enable_weapon_detection: bool = True, weapon_confidence_threshold: float = 0.2):
        self.logger = logging.getLogger(__name__)
        self.model = YOLO(model_path)
        self.person_confidence_threshold = person_confidence_threshold
        self.person_class_id = PERSON_CLASS_ID
        
        # Initialize weapon detector if available and enabled
        self.weapon_detector = None
        self.enable_weapon_detection = enable_weapon_detection and WEAPON_DETECTION_AVAILABLE
        self.weapon_confidence_threshold = weapon_confidence_threshold
        
        if self.enable_weapon_detection:
            try:
                self.weapon_detector = WeaponDetector(confidence_threshold=weapon_confidence_threshold)
                self.logger.info("Weapon detection enabled (confidence: %s)", weapon_confidence_threshold)
            except Exception as e:
                self.logger.warning("Failed to initialize weapon detector: %s", e)
                self.enable_weapon_detection = False
        else:
            if enable_weapon_detection and not WEAPON_DETECTION_AVAILABLE:
                self.logger.info("Weapon detection requested but not available")
            else:
                self.logger.info("Weapon detection disabled")
        
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
            self.logger.warning("Failed to initialize distance estimation: %s", e)

    @classmethod
    def extract_filename_metadata(cls, path: str):
        """Extract (class, distance, height, camera pitch) from a path or sample name.

        Expected naming convention somewhere in the path:
            (real|falso)_<distance>_<height>_<camera_pitch>

        Examples:
            real_05_02_10.MP4
            falso_05_02_10_clip_000_10s/frame_0000.jpg

        Returns a dict with keys:
            sample_class: 'real'|'falso'|None
            distance_m: float|None
            height_m: float|None
            camera_pitch_deg: float|None
        """
        if path is None:
            return {'sample_class': None, 'distance_m': None, 'height_m': None, 'camera_pitch_deg': None}

        # Search from the most specific path component outward.
        p = PurePath(str(path))
        candidates = []

        # Full name (file or directory)
        try:
            candidates.append(p.name)
        except Exception:
            pass

        # Parent folder name (often the sample directory)
        try:
            if p.parent is not None:
                candidates.append(p.parent.name)
        except Exception:
            pass

        # Also scan every component (reverse) in case of deeper nesting
        try:
            candidates.extend(reversed(p.parts))
        except Exception:
            pass

        for text in candidates:
            if not text:
                continue
            m = cls._FILENAME_META_RE.search(str(text))
            if not m:
                continue
            sample_class = m.group(1).lower()
            try:
                distance_m = float(m.group(2))
            except Exception:
                distance_m = None
            try:
                height_m = float(m.group(3))
            except Exception:
                height_m = None
            try:
                camera_pitch_deg = float(m.group(4))
            except Exception:
                camera_pitch_deg = None
            return {
                'sample_class': sample_class,
                'distance_m': distance_m,
                'height_m': height_m,
                'camera_pitch_deg': camera_pitch_deg,
            }

        return {'sample_class': None, 'distance_m': None, 'height_m': None, 'camera_pitch_deg': None}
    
    def extract_real_distance_from_filename(self, filepath: str):
        """Backward-compatible wrapper for filename distance extraction."""
        return self.extract_filename_metadata(filepath).get('distance_m')
        
    def extract_camera_height_from_filename(self, filepath: str):
        """Backward-compatible wrapper for filename camera height extraction."""
        return self.extract_filename_metadata(filepath).get('height_m')

    def detect_people(self, image_path: str, draw_boxes: bool = False):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Run inference - only detect person class (class 0)
        results = self.model(image, imgsz=640, iou=0.6, conf=self.person_confidence_threshold, classes=[0], verbose=False)
        
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
                    if class_id == self.person_class_id and confidence >= self.person_confidence_threshold:
                        # height pixels
                        person_height_px = y2 - y1
                        
                        distance_m = None  # primary distance used downstream
                        distance_pinhole_m = None
                        distance_pitch_m = None
                        distance_method = None
                        bearing_deg = None
                        metadata = None
                        has_metadata = False
                        if self.camera:
                            from position_estimation import estimate_distance, estimate_distance_2, estimate_bearing
                            try:
                                image_name = os.path.basename(image_path)
                                # Try to load metadata from JSON file first
                                has_metadata = self.camera.load_from_json(image_path)
                                # Testing-time condition metadata from filename (kept as annotated values).
                                filename_meta = self.extract_filename_metadata(image_path)
                                height_annotated_m = filename_meta.get('height_m')
                                if has_metadata:
                                    camera_height_m = self.camera.camera_height_m
                                    camera_pitch_deg = self.camera.pitch_deg
                                    real_distance_m = None  # No ground truth in metadata
                                    y_bottom = y2
                                    distance_pitch_m = estimate_distance_2(self.camera, y_bottom)
                                    distance_pinhole_m = estimate_distance(self.camera, person_height_px) if person_height_px > 0 else None
                                    x_center = float((x1 + x2) / 2)
                                    bearing_deg = float(estimate_bearing(self.camera, x_center))
                                else:
                                    real_distance_m = filename_meta.get('distance_m')
                                    # No JSON -> we do NOT have real camera height/pitch; keep annotated height for metrics only.
                                    camera_height_m = None
                                    # Camera pitch from filename convention (testing-time condition / dataset parameter).
                                    camera_pitch_deg = filename_meta.get('camera_pitch_deg')
                                    # Always compute pinhole estimate.
                                    distance_pinhole_m = estimate_distance(self.camera, person_height_px) if person_height_px > 0 else None

                                    # Pitch-based estimate is only valid when real camera height + pitch are available.
                                    distance_pitch_m = None

                                    # Even without metadata, we can still estimate bearing from pixel x and current yaw.
                                    x_center = float((x1 + x2) / 2)
                                    bearing_deg = float(estimate_bearing(self.camera, x_center))

                                # Choose primary distance for downstream use.
                                if distance_pitch_m is not None:
                                    distance_m = distance_pitch_m
                                    distance_method = 'pitch'
                                elif distance_pinhole_m is not None:
                                    distance_m = distance_pinhole_m
                                    distance_method = 'pinhole'
                                else:
                                    distance_m = None
                                    distance_method = 'none'
                                primary_str = f"{distance_m:.2f}" if distance_m is not None else "N/A"
                                pinhole_str = f"{distance_pinhole_m:.2f}" if distance_pinhole_m is not None else "N/A"
                                pitch_str = f"{distance_pitch_m:.2f}" if distance_pitch_m is not None else "N/A"

                                # Build log message with proper formatting
                                if has_metadata:
                                    log_message = (f"Image: {image_name}, Person: {person_idx + 1}, "
                                                 f"PixelHeight: {person_height_px:.1f}px, "
                                                 f"Estimated({distance_method}): {primary_str}m, "
                                                 f"Pinhole: {pinhole_str}m, PitchBased: {pitch_str}m, "
                                                 f"GPS: ({self.camera.lat:.6f}, {self.camera.lon:.6f}), "
                                                 f"CameraHeight(real): {camera_height_m:.2f}m, "
                                                 f"Pitch: {camera_pitch_deg:.2f}deg, "
                                                 f"Yaw: {self.camera.yaw_deg:.2f}deg, "
                                                 f"Confidence: {confidence:.3f}, "
                                                 f"BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                                    
                                    # Enhanced console output with metadata
                                    console_msg = f"  -> Person {person_idx + 1}: Est({distance_method}):{primary_str}m"
                                    console_msg += f", HeightReal:{camera_height_m:.2f}m, Pitch:{camera_pitch_deg:.0f}deg"
                                    console_msg += f", {person_height_px:.1f}px (metadata)"
                                else:
                                    # Legacy mode logging
                                    real_dist_str = f"{real_distance_m:.2f}" if real_distance_m is not None else "N/A"
                                    cam_height_str = f"{camera_height_m:.2f}" if camera_height_m is not None else "N/A"
                                    height_ann_str = f"{height_annotated_m:.2f}" if height_annotated_m is not None else "N/A"
                                    cam_pitch_str = f"{camera_pitch_deg:.2f}" if camera_pitch_deg is not None else "N/A"
                                    log_message = (f"Image: {image_name}, Person: {person_idx + 1}, "
                                                 f"PixelHeight: {person_height_px:.1f}px, "
                                                 f"Estimated({distance_method}): {primary_str}m, "
                                                 f"Pinhole: {pinhole_str}m, PitchBased: {pitch_str}m, "
                                                 f"Real: {real_dist_str}m, "
                                                 f"CameraHeight(real): {cam_height_str}m, "
                                                 f"HeightAnnotated: {height_ann_str}m, "
                                                 f"CameraPitch(annotated): {cam_pitch_str}deg, "
                                                 f"Confidence: {confidence:.3f}, "
                                                 f"BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                                    
                                    # Legacy console output
                                    console_msg = f"  -> Person {person_idx + 1}: Est({distance_method}):{primary_str}m"
                                    if real_distance_m is not None:
                                        console_msg += f", Real:{real_distance_m:.2f}m"
                                    if height_annotated_m is not None:
                                        console_msg += f", HeightAnn:{height_annotated_m:.2f}m"
                                    if camera_pitch_deg is not None:
                                        console_msg += f", PitchAnn:{camera_pitch_deg:.0f}deg"
                                    console_msg += f", {person_height_px:.1f}px"
                                
                                self.distance_logger.info(log_message)
                                
                            except Exception as e:
                                self.logger.warning("Failed to estimate distance for person %s: %s", person_idx + 1, e)
                        
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
                            'person_confidence': float(confidence),
                            'class': 'person',
                            'height_px': float(person_height_px)
                        }

                        # Provide both real and annotated camera parameters.
                        # - "real" values come from drone metadata JSON (when available)
                        # - "annotated" values come from filename convention (testing-time condition)
                        if height_annotated_m is not None:
                            detection_info['camera_height_annotated_m'] = float(height_annotated_m)
                        try:
                            pitch_ann = filename_meta.get('camera_pitch_deg') if 'filename_meta' in locals() else None
                        except Exception:
                            pitch_ann = None
                        if pitch_ann is not None:
                            detection_info['camera_pitch_annotated_deg'] = float(pitch_ann)
                        if has_metadata:
                            detection_info['camera_height_real_m'] = float(self.camera.camera_height_m)
                            detection_info['camera_pitch_real_deg'] = float(self.camera.pitch_deg)
                            detection_info['camera_yaw_real_deg'] = float(self.camera.yaw_deg)
                        
                        # Add distance if available
                        if distance_m is not None:
                            detection_info['distance_m'] = float(distance_m)
                            detection_info['distance_method'] = distance_method

                        # Always record both estimates when available (for evaluation).
                        if distance_pinhole_m is not None:
                            detection_info['distance_pinhole_m'] = float(distance_pinhole_m)
                        if distance_pitch_m is not None:
                            detection_info['distance_pitch_m'] = float(distance_pitch_m)

                        # Add bearing if available
                        if bearing_deg is not None:
                            detection_info['bearing_deg'] = float(bearing_deg)
                        
                        # Add metadata if available
                        if has_metadata:
                            # Calculate person's geographic position (for comparison/consistency with dual-drone).
                            person_lat, person_lon = None, None
                            if distance_m is not None and distance_m > 0:
                                try:
                                    x_center = float((x1 + x2) / 2)
                                    person_lat, person_lon = self.camera.calculate_person_geoposition(
                                        camera_lat=self.camera.lat,
                                        camera_lon=self.camera.lon,
                                        camera_yaw_deg=self.camera.yaw_deg,
                                        x_pixel=x_center,
                                        distance_m=distance_m
                                    )
                                except Exception as e:
                                    self.logger.warning("Failed to calculate person geoposition: %s", e)
                            
                            detection_info['metadata'] = {
                                'gps': {
                                    'latitude': self.camera.lat,
                                    'longitude': self.camera.lon,
                                        'camera_height_real_m': self.camera.camera_height_m
                                },
                                'orientation': {
                                    'yaw': self.camera.yaw_deg,
                                    'pitch': self.camera.pitch_deg
                                },
                                'timestamp': self.camera.timestamp_sec,
                                'datetime': self.camera.datetime
                            }

                            # Convenience alias for downstream code that prefers "height" naming.
                            detection_info['metadata']['gps']['camera_height_real_m'] = self.camera.camera_height_m
                            
                            # Add person's estimated geographic position if calculated
                            if person_lat is not None and person_lon is not None:
                                detection_info['person_geoposition'] = {
                                    'latitude': person_lat,
                                    'longitude': person_lon,
                                    'bearing_deg': bearing_deg
                                }
                        
                        detections_info.append(detection_info)
        
        return image_with_boxes, detections_info
