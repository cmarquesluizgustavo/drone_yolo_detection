import cv2
import os
import logging
from ultralytics import YOLO
from camera import Camera

from geoconverter import GeoConverter
from position_estimation import estimate_distance, estimate_distance_pitch, estimate_bearing, estimate_distance_fused

# Detection and visualization constants
PERSON_CLASS_ID = 0  # COCO class ID for 'person'
BOX_COLOR = (0, 255, 0)  # Green bounding boxes (BGR format)
BOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 2
CROP_PADDING = 0.1  # 10% padding around bounding boxes
CROP_MIN_SIZE = 32  # Minimum crop size in pixels

try:
    from pipeline.weapon_detector import WeaponDetector
    WEAPON_DETECTION_AVAILABLE = True
except ImportError as e:
    WEAPON_DETECTION_AVAILABLE = False
    logging.getLogger(__name__).warning("Weapon detection not available: %s", e)


class PeopleDetector:
    """Core detector class for person and weapon detection in single images."""
    
    def __init__(
        self,
        model_path: str,
        person_confidence_threshold: float = 0.5,
        enable_weapon_detection: bool = True,
        weapon_confidence_threshold: float = 0.5,
        device=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.model = YOLO(model_path)
        self.device = device
        self.person_confidence_threshold = person_confidence_threshold
        self.person_class_id = PERSON_CLASS_ID
        
        # Initialize weapon detector if available and enabled
        self.weapon_detector = None
        self.enable_weapon_detection = enable_weapon_detection and WEAPON_DETECTION_AVAILABLE
        self.weapon_confidence_threshold = weapon_confidence_threshold
        
        if self.enable_weapon_detection:
            try:
                self.weapon_detector = WeaponDetector(
                    weapon_confidence_threshold=weapon_confidence_threshold,
                    device=device,
                )
                self.logger.info("Weapon detection enabled (WEAPON_CONFIDENCE=%s)", weapon_confidence_threshold)
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
        except Exception as e:
            self.logger.warning("Failed to initialize distance estimation: %s", e)

    def extract_filename_metadata(self, path: str):
        """Extract metadata from filename pattern: (real|falso)_<distance>_<height>"""
        import re
        import os
        
        basename = os.path.basename(path)

        # Pattern: (real|falso)_<distance>_<height>
        pattern = r'(real|falso)_(\d+)_(\d+)'
        match = re.search(pattern, basename, re.IGNORECASE)
        if not match:
            # Frame files are typically named frame_0001.jpg; the metadata is stored in
            # the sample folder name (e.g., falso_05_02_clip_...).
            match = re.search(pattern, str(path), re.IGNORECASE)
        
        if match:
            sample_class = match.group(1).lower()
            distance_m = float(match.group(2))
            height_m = float(match.group(3))
            
            return {
                'sample_class': sample_class,
                'distance_m': distance_m,
                'height_m': height_m,
            }
        
        return {}

    def detect_people(self, image_path: str, draw_boxes: bool = False):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Run inference - only detect person class (class 0)
        infer_kwargs = dict(
            imgsz=640,
            iou=0.6,
            conf=self.person_confidence_threshold,
            classes=[0],
            verbose=False,
        )
        if self.device is not None:
            infer_kwargs["device"] = self.device

        results = self.model(image, **infer_kwargs)
        
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
                        bearing_deg = None
                        if self.camera:
                            try:
                                image_name = os.path.basename(image_path)
                                # Get annotated values from filename ONLY
                                file_data = self.extract_filename_metadata(image_path)
                                self.camera.height_m = file_data.get('height_m')
                                real_distance_m = file_data.get('distance_m')

                                # height based x pitch based
                                distance_pinhole_m = estimate_distance(self.camera, person_height_px)
                                distance_pitch_m = estimate_distance_pitch(self.camera, y2)
                                distance_fused_m = estimate_distance_fused(self.camera, person_height_px, y2)

                                x_center = float((x1 + x2) / 2)
                                bearing_deg = float(estimate_bearing(self.camera, x_center))

                                pinhole_str = f"{distance_pinhole_m:.2f}"
                                pitch_str = f"{distance_pitch_m:.2f}"

                                # Build log message with annotated values
                                real_dist_str = f"{real_distance_m:.2f}"
                                cam_height_str = f"{self.camera.height_m:.2f}"
                                cam_pitch_str = f"{self.camera.pitch_deg:.2f}"
                                log_message = (f"Image: {image_name}, Person: {person_idx + 1}, "
                                             f"PixelHeight: {person_height_px:.1f}px, "
                                             f"HeightBased: {pinhole_str}m, PitchBased: {pitch_str}m, "
                                             f"Fused: {distance_fused_m:.2f}m, "
                                             f"Real: {real_dist_str}m, "
                                             f"CamHeight: {cam_height_str}m, "
                                             f"CamPitch: {cam_pitch_str}deg, "
                                             f"Confidence: {confidence:.3f}, "
                                             f"BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                                
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

                        # Add annotated values from filename
                        if self.camera.height_m is not None:
                            detection_info['camera_height_m'] = float(self.camera.height_m)
                        if self.camera.pitch_deg is not None:
                            detection_info['camera_pitch_deg'] = float(self.camera.pitch_deg)

                        # Always record both estimates when available (for evaluation).
                        if distance_pinhole_m is not None:
                            detection_info['distance_pinhole_m'] = float(distance_pinhole_m)
                        if distance_pitch_m is not None:
                            detection_info['distance_pitch_m'] = float(distance_pitch_m)
                        if distance_fused_m is not None:
                            detection_info['distance_fused_m'] = float(distance_fused_m)

                        # Add bearing if available
                        if bearing_deg is not None:
                            detection_info['bearing_deg'] = float(bearing_deg)
                        
                        detections_info.append(detection_info)
        
        return image_with_boxes, detections_info
    
    def detect_people_with_estimation(self, image_path, image):
        infer_kwargs = dict(
            imgsz=640,
            iou=0.6,
            conf=self.person_confidence_threshold,
            classes=[0],
            verbose=False,
        )
        if getattr(self, "device", None) is not None:
            infer_kwargs["device"] = getattr(self, "device")
        results = self.model(image, **infer_kwargs)
        file_data = self.extract_filename_metadata(image_path)

        # Set camera height and pitch from metadata if available
        cam_height_m = file_data.get('height_m') if file_data.get('height_m') is not None else file_data.get('cam_height_m')
        cam_pitch_deg = file_data.get('pitch_deg') if file_data.get('pitch_deg') is not None else file_data.get('cam_pitch_deg')
        if cam_height_m is not None:
            self.camera.height_m = cam_height_m
        if cam_pitch_deg is not None:
            self.camera.pitch_deg = cam_pitch_deg

        detections_info = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id == 0 and confidence >= self.person_confidence_threshold:
                        person_height_px = float(y2 - y1)
                        y_bottom = float(y2)
                        x_center = float((x1 + x2) / 2)

                        # Always compute pinhole estimate
                        distance_pinhole_m = estimate_distance(self.camera, person_height_px) if person_height_px > 0 else None

                        # Compute pitch-based estimate if we have annotated height + pitch
                        distance_pitch_m = None
                        if self.camera.height_m is not None and self.camera.pitch_deg is not None:
                            distance_pitch_m = estimate_distance_pitch(self.camera, y_bottom)

                        distance_fused_m = estimate_distance_fused(self.camera, person_height_px, y_bottom)

                        # Choose primary distance for downstream use
                        if distance_fused_m is not None:
                            distance_m = distance_fused_m
                            distance_method = 'fused'
                        elif distance_pitch_m is not None:
                            distance_m = distance_pitch_m
                            distance_method = 'pitch'
                        elif distance_pinhole_m is not None:
                            distance_m = distance_pinhole_m
                            distance_method = 'pinhole'
                        else:
                            distance_m = None
                            distance_method = 'none'

                        bearing_deg = estimate_bearing(self.camera, x_center)

                        # Compute geoposition if possible
                        person_geoposition = None
                        if self.camera.lat is not None and self.camera.lon is not None and distance_m is not None and bearing_deg is not None:
                            try:
                                # print(f"[GeoEstimation DEBUG] lat={self.camera.lat}, lon={self.camera.lon}, distance_m={distance_m}, bearing_deg={bearing_deg}")
                                # Use GeoConverter to compute new lat/lon from current position, distance, and bearing (polar)
                                latlon = GeoConverter.polar_to_geo(
                                    self.camera.lat, self.camera.lon, bearing_deg, distance_m
                                )
                                if latlon and latlon[0] is not None and latlon[1] is not None:
                                    person_geoposition = {"latitude": latlon[0], "longitude": latlon[1]}
                                else:
                                    person_geoposition = None
                            except Exception as e:
                                self.logger.warning("Failed to compute geoposition: %s", e)

                        detections_info.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'person_confidence': confidence,
                            'distance_m': distance_m,
                            'distance_method': distance_method,
                            'distance_pinhole_m': distance_pinhole_m,
                            'distance_fused_m': distance_fused_m,
                            'distance_pitch_m': distance_pitch_m,
                            'bearing_deg': bearing_deg,
                            'cam_height_m': self.camera.height_m,
                            'cam_pitch_deg': self.camera.pitch_deg,
                            'person_geoposition': person_geoposition,
                        })
        return detections_info
