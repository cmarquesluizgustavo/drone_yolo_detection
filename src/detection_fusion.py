import math
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    person_confidence: float
    distance_m: float
    bearing_deg: float
    x: float  # Ground-plane X coordinate (meters)
    y: float  # Ground-plane Y coordinate (meters)
    lat: float
    lon: float
    drone_id: int
    frame_id: int
    weapon_confidence: float
    has_weapon: bool = False


class DualDroneFusion:
    """Handles fusion of detections from two drones."""

    def __init__(self, association_threshold_m, weapon_threshold=0.5):
        # maximum distance between detections to be considered the same target
        self.association_threshold_m = association_threshold_m
        self.weapon_threshold = weapon_threshold
        
    def fuse_confidence(self, conf1, conf2):
        """probabilistic evidence accumulation!"""
        # c_f = 1 - (1-c1)^w1 * (1-c2)^w2
        return 1 - (1 - conf1) * (1 - conf2)
    
    def detection_to_dict(self, det):
        """Convert a Detection dataclass object to a dictionary."""
        return {
            'bbox': det.bbox,
            'person_confidence': det.person_confidence,
            'distance_m': det.distance_m,
            'bearing_deg': det.bearing_deg,
            'x': det.x,
            'y': det.y,
            'lat': det.lat,
            'lon': det.lon,
            'has_weapon': det.has_weapon,
            'weapon_confidence': det.weapon_confidence,
            'source': f'drone{det.drone_id}',
            'drone_id': det.drone_id,
            'frame_id': det.frame_id
        }

    def match_detections(self, detections1, detections2):
        """Associate detections from two drones based on ground-plane proximity."""
        fused_detections = []
        used_det2 = set()
        
        for det1 in detections1:
            best_match = None
            best_distance = float('inf')

            # If det1 has no valid ground-plane coordinates, don't attempt cross-drone association.
            if not (math.isfinite(det1.x) and math.isfinite(det1.y)):
                fused_detections.append(self.detection_to_dict(det1))
                continue
            
            # Find closest detection from drone 2
            for idx2, det2 in enumerate(detections2):
                if idx2 in used_det2:
                    continue

                # Skip detections without valid coordinates.
                if not (math.isfinite(det2.x) and math.isfinite(det2.y)):
                    continue
                
                # Compute ground-plane distance
                dx = det1.x - det2.x
                dy = det1.y - det2.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < self.association_threshold_m and distance < best_distance:
                    best_match = (idx2, det2)
                    best_distance = distance
            
            if best_match is not None:
                # Found a match - fuse the detections
                idx2, det2 = best_match
                used_det2.add(idx2)
                
                fused = self.fuse_detections(det1, det2)
                fused_detections.append(fused)
            else:
                # No match - keep detection from drone 1 only
                fused_detections.append(self.detection_to_dict(det1))
        
        # Add unmatched detections from drone 2
        for idx2, det2 in enumerate(detections2):
            if idx2 not in used_det2:
                fused_detections.append(self.detection_to_dict(det2))
        
        return fused_detections
    
    def fuse_detections(self, det1, det2):
        """Fuse two detections using probabilistic confidence fusion."""
        fused_conf = self.fuse_confidence(det1.person_confidence, det2.person_confidence)
        fused_weapon_conf = self.fuse_confidence(det1.weapon_confidence, det2.weapon_confidence)
        has_weapon = fused_weapon_conf > self.weapon_threshold
        
        return {
            'person_confidence': fused_conf,
            'x': 0.0,
            'y': 0.0,
            'lat': 0.0,
            'lon': 0.0,
            'has_weapon': has_weapon,
            'weapon_confidence': fused_weapon_conf,
            'source': 'fused',
            'drone_ids': [det1.drone_id, det2.drone_id],
            'frame_id': det1.frame_id,
            'bbox_drone1': det1.bbox,
            'bbox_drone2': det2.bbox
        }