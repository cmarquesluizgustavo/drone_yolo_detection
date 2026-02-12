import logging
from pathlib import Path
import os

import cv2
from ultralytics import YOLO


class WeaponDetector:
    def __init__(self, model_path: str = None, weapon_confidence_threshold: float = 0.5, device=None):
        if model_path is None:
            current_dir = Path(__file__).parent
            model_path = current_dir.parent / "models" / "weapons" / "best.pt"

        self.model_path = model_path
        self.weapon_confidence_threshold = weapon_confidence_threshold
        self.weapon_class_id = 0
        self.logger = logging.getLogger(__name__)
        self.device = device

        try:
            self.model = YOLO(str(self.model_path))
            self.logger.info("Loaded weapon detection model: %s", self.model_path)
        except Exception as e:
            # Common on Windows when a .pt was created on Linux and contains pathlib.PosixPath objects.
            # Unpickling then fails with: "cannot instantiate 'PosixPath' on your system".
            if os.name == "nt" and "PosixPath" in str(e) and "cannot instantiate" in str(e):
                try:
                    import pathlib as _pathlib

                    self.logger.warning(
                        "Retrying weapon model load with PosixPath->WindowsPath patch (Windows compatibility workaround)."
                    )
                    _pathlib.PosixPath = _pathlib.WindowsPath  # type: ignore[attr-defined]
                    self.model = YOLO(str(self.model_path))
                    self.logger.info("Loaded weapon detection model (Windows path fix applied): %s", self.model_path)
                except Exception as e2:
                    self.logger.warning("Error loading weapon detection model after path fix: %s", e2)
                    raise RuntimeError(f"Failed to load weapon detection model: {e2}")
            else:
                self.logger.warning("Error loading weapon detection model: %s", e)
                raise RuntimeError(f"Failed to load weapon detection model: {e}")

    def detect_weapons(self, image):
        try:
            infer_kwargs = dict(conf=self.weapon_confidence_threshold, iou=0.4, verbose=False)
            if self.device is not None:
                infer_kwargs["device"] = self.device
            results = self.model(image, **infer_kwargs)

            annotated_image = image.copy()
            weapon_crops = []
            detections_info = []

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                    weapon_confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.model.names[class_id]

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{class_name}: {weapon_confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    padding = 5
                    x1_pad = max(0, x1 - padding)
                    y1_pad = max(0, y1 - padding)
                    x2_pad = min(image.shape[1], x2 + padding)
                    y2_pad = min(image.shape[0], y2 + padding)

                    weapon_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                    if weapon_crop.size > 0:
                        weapon_crops.append(
                            {
                                'crop': weapon_crop,
                                'bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
                                'weapon_confidence': weapon_confidence,
                                'class': class_name,
                            }
                        )

                    detections_info.append(
                        {
                            'bbox': [x1, y1, x2, y2],
                            'weapon_confidence': weapon_confidence,
                            'class': class_name,
                        }
                    )

            return annotated_image, detections_info, weapon_crops
        except Exception as e:
            self.logger.warning("Error in weapon detection: %s", e)
            return image, [], []

    def process_person_crop(self, crop_image, crop_info):
        annotated_crop, weapon_detections, weapon_crops = self.detect_weapons(crop_image)
        return {
            'original_crop': crop_image,
            'annotated_crop': annotated_crop,
            'person_info': crop_info,
            'weapon_detections': weapon_detections,
            'weapon_crops': weapon_crops,
            'has_weapon': len(weapon_detections) > 0,
        }

    def process_multiple_crops(self, crops_with_info):
        return [self.process_person_crop(crop_image, crop_info) for crop_image, crop_info in crops_with_info]
