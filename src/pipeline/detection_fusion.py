import math
from typing import Tuple
from dataclasses import dataclass
from geoconverter import GeoConverter

@dataclass
class Detection:
    """Detection from a single drone."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    person_confidence: float
    weapon_confidence: float
    has_weapon: bool
    
    distance_m: float # do drone pro target
    bearing_deg: float # do drone pro tgarget
    
    # coordenadas estimadas pro target
    x: float
    y: float
    lat: float 
    lon: float 

    drone_id: int # qual drone
    frame_id: int # qual frame
    


class DualDroneFusion:
    def _compute_iou(self, bbox1, bbox2):
        # bbox: (x1, y1, x2, y2)
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area
    def single_track(self, det, uav_pos, drone_name):
        return {
            'drone_measurements': [{
                'uav_pos': uav_pos,
                'distance': det.distance_m,
                'bearing': det.bearing_deg,
                'person_confidence': det.person_confidence
            }],
            'person_confidence': det.person_confidence,
            'has_weapon': det.has_weapon,
            'weapon_confidence': det.weapon_confidence,
            f'bbox_{drone_name}': det.bbox,
            'drone_id': det.drone_id,
            'frame_id': det.frame_id
        }

    def __init__(self, association_threshold_m, weapon_threshold):
        self.association_threshold_m = association_threshold_m
        self.weapon_threshold = weapon_threshold
        
    def fuse_confidence(self, conf1, conf2):
        return 1.0 - (1.0 - conf1) * (1.0 - conf2)

    def fuse_confidence_2(self, conf1, conf2):
        return (conf1 + conf2) / 2.0

    def fuse_confidence_3(self, conf1, conf2):
        return max(conf1, conf2)

    def matching(self, detections1, detections2, camera1, camera2):
        # Get UAV positions in ground plane coordinates
        uav1_x, uav1_y = GeoConverter.geo_to_xy(camera1.lat, camera1.lon)
        uav2_x, uav2_y = GeoConverter.geo_to_xy(camera2.lat, camera2.lon)
        
        measurement_groups = []
        used_det2 = set()
        
        # drone1 processing
        for idx1, det1 in enumerate(detections1):
            # matching apartir da distancia euclidiana dos pontos
            best_match_idx = None
            best_distance = self.association_threshold_m
            for idx2, det2 in enumerate(detections2):
                if idx2 in used_det2:
                    continue
                dx = det1.x - det2.x
                dy = det1.y - det2.y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < best_distance:
                    best_match_idx = idx2
                    best_distance = distance
            if best_match_idx is not None:
                det2 = detections2[best_match_idx]
                measurement_groups.append(self.fused_track(det1, det2, (uav1_x, uav1_y), (uav2_x, uav2_y)))
                used_det2.add(best_match_idx)
            else:
                measurement_groups.append(self.single_track(det1, (uav1_x, uav1_y), 'drone1'))
        # Add unmatched detections from drone2
        for idx2, det2 in enumerate(detections2):
            if idx2 not in used_det2:
                measurement_groups.append(self.single_track(det2, (uav2_x, uav2_y), 'drone2'))
        return measurement_groups
    
    def fused_track(self, det1, det2, uav1_pos, uav2_pos):
        # Fuse confidences
        fused_person_conf = self.fuse_confidence(
            det1.person_confidence, det2.person_confidence
        )
        fused_weapon_conf = self.fuse_confidence(
            det1.weapon_confidence, det2.weapon_confidence
        )
        
        return {
            'drone_measurements': [
                {
                    'uav_pos': uav1_pos,
                    'distance': det1.distance_m,
                    'bearing': det1.bearing_deg,
                    'person_confidence': det1.person_confidence,
                },
                {
                    'uav_pos': uav2_pos,
                    'distance': det2.distance_m,
                    'bearing': det2.bearing_deg,
                    'person_confidence': det2.person_confidence
                }
            ],
            'person_confidence': fused_person_conf,
            'has_weapon': fused_weapon_conf >= self.weapon_threshold,
            'weapon_confidence': fused_weapon_conf,
            'bbox_drone1': det1.bbox,
            'bbox_drone2': det2.bbox,
            'drone_ids': [det1.drone_id, det2.drone_id],
            'frame_id': det1.frame_id
        }
