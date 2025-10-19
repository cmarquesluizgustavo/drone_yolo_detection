"""
Weapon Detection using YOLOv8 Model
Detects weapons in cropped person images.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

class WeaponDetector:
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.2):
        """
        Initialize the weapon detector with YOLOv8 model.
        
        Args:
            model_path: Path to the YOLOv8 weapon detection model
            confidence_threshold: Minimum confidence for detections
        """
        if model_path is None:
            # Default path to the weapons model
            current_dir = Path(__file__).parent
            model_path = current_dir.parent / "models" / "weapons" / "yolov8guns.pt"
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.weapon_class_id = 0  # Based on the model output, gun class is ID 0
        
        # Load the model
        try:
            self.model = YOLO(str(self.model_path))
            print(f"Loaded YOLOv8 weapon detection model: {self.model_path}")
            print(f"Model classes: {self.model.names}")
        except Exception as e:
            raise RuntimeError(f"Failed to load weapon detection model: {e}")
    
    def detect_weapons(self, image):
        """
        Detect weapons in a single image using YOLOv8.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            tuple: (annotated_image, detections_info, weapon_crops)
        """
        try:
            # Run YOLOv8 inference with IOU threshold for better NMS
            # iou=0.4 means boxes with IoU > 0.4 will be suppressed (keeps only best box)
            results = self.model(image, conf=self.confidence_threshold, iou=0.4, verbose=False)
            
            # Create annotated image
            annotated_image = image.copy()
            
            # Extract weapon crops and detection info
            weapon_crops = []
            detections_info = []
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get detections
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Get bounding box coordinates (xyxy format)
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Get confidence score
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Get class (should be 'gun' for this model)
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Draw bounding box and label on annotated image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 0, 255), -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Add small padding to weapon bbox for cropping
                    padding = 5
                    x1_pad = max(0, x1 - padding)
                    y1_pad = max(0, y1 - padding)
                    x2_pad = min(image.shape[1], x2 + padding)
                    y2_pad = min(image.shape[0], y2 + padding)
                    
                    # Extract weapon crop
                    weapon_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    if weapon_crop.size > 0:  # Ensure crop is valid
                        weapon_crops.append({
                            'crop': weapon_crop,
                            'bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
                            'confidence': confidence,
                            'class': class_name
                        })
                    
                    # Store detection info
                    detections_info.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_name
                    })
            
            return annotated_image, detections_info, weapon_crops
            
        except Exception as e:
            print(f"Error in weapon detection: {e}")
            return image, [], []
    
    def process_person_crop(self, crop_image, crop_info):
        """
        Process a single person crop for weapon detection.
        
        Args:
            crop_image: Cropped person image (numpy array)
            crop_info: Information about the person crop
            
        Returns:
            dict: Results containing annotated image and detection info
        """
        annotated_crop, weapon_detections, weapon_crops = self.detect_weapons(crop_image)
        
        return {
            'original_crop': crop_image,
            'annotated_crop': annotated_crop,
            'person_info': crop_info,
            'weapon_detections': weapon_detections,
            'weapon_crops': weapon_crops,
            'has_weapons': len(weapon_detections) > 0
        }
    
    def process_multiple_crops(self, crops_with_info):
        """
        Process multiple person crops for weapon detection.
        
        Args:
            crops_with_info: List of tuples (crop_image, crop_info)
            
        Returns:
            list: List of detection results for each crop
        """
        results = []
        
        for crop_image, crop_info in crops_with_info:
            result = self.process_person_crop(crop_image, crop_info)
            results.append(result)
            
            # Log detection results
            # if result['has_weapons']:
            #     weapon_count = len(result['weapon_detections'])
            #     print(f"  -> Weapons detected in person crop: {weapon_count} weapon(s)")
            # else:
            #     print(f"  -> No weapons detected in person crop")
        
        return results
