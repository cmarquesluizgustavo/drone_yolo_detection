import math
from typing import Dict, Tuple
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
    has_weapon: bool = False
    weapon_confidence: float = 0.0


class DualDroneFusion:
    """Handles fusion of detections from two drones."""

    def __init__(self, association_threshold_m = 2.0):
        # maximum distance between detections to be considered the same target
        self.association_threshold_m = association_threshold_m
        
    def fuse_confidence(self, conf1, conf2):
        """probabilistic evidence accumulation!"""
        # Weighted version: c_f = 1 - (1-c1)^w1 * (1-c2)^w2
        return 1 - (1 - conf1) * (1 - conf2)

    def match_detections(self, detections1, detections2):
        """Associate detections from two drones based on ground-plane proximity."""
        fused_detections = []
        used_det2 = set()
        
        for det1 in detections1:
            best_match = None
            best_distance = float('inf')
            
            # Find closest detection from drone 2
            for idx2, det2 in enumerate(detections2):
                if idx2 in used_det2:
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
    
    def fuse_detections(self, det1, det2, weapon_threshold=0.2):
        """Fuse a pair of matched detections."""
        # Fuse confidence
        fused_conf = self.fuse_confidence(det1.person_confidence, det2.person_confidence)
        fused_weapon_conf = self.fuse_confidence(det1.weapon_confidence, det2.weapon_confidence)
        has_weapon = fused_weapon_conf > weapon_threshold
        
        # Average ground position
        fused_x = (det1.x + det2.x) / 2.0
        fused_y = (det1.y + det2.y) / 2.0
        
        # Average GPS (simple approach)
        fused_lat = (det1.lat + det2.lat) / 2.0
        fused_lon = (det1.lon + det2.lon) / 2.0
        
        return {
            'person_confidence': fused_conf,
            'x': fused_x,
            'y': fused_y,
            'lat': fused_lat,
            'lon': fused_lon,
            'has_weapon': has_weapon,
            'weapon_confidence': fused_weapon_conf,
            'source': 'fused',
            'drone_ids': [det1.drone_id, det2.drone_id],
            'frame_id': det1.frame_id,
            'bbox_drone1': det1.bbox,
            'bbox_drone2': det2.bbox
        }
    
    def detection_to_dict(self, det):
        """Convert a single Detection to dictionary format."""
        return {
            'person_confidence': det.person_confidence,
            'x': det.x,
            'y': det.y,
            'lat': det.lat,
            'lon': det.lon,
            'has_weapon': det.has_weapon,
            'weapon_confidence': det.weapon_confidence,
            'source': f'drone{det.drone_id}',
            'drone_ids': [det.drone_id],
            'frame_id': det.frame_id,
            'bbox': det.bbox
        }


class FrameSynchronizer:

    def __init__(self, max_time_diff_ms=100):
        # maximum time difference (milliseconds) to consider frames synchronized
        self.max_time_diff = max_time_diff_ms

    def synchronize_by_frame_index(self, frames1, frames2):
        # Extract frame numbers from filenames
        def extract_frame_num(path: str) -> int:
            import re
            # Look for patterns like _0000 or frame_000 in filename
            basename = path.split('/')[-1].split('\\')[-1]
            matches = re.findall(r'_(\d{4})', basename)
            if matches:
                return int(matches[-1])  # Take last match
            matches = re.findall(r'(\d{4})', basename)
            if matches:
                return int(matches[-1])
            return -1
        
        # Build frame index maps
        frames1_dict = {extract_frame_num(f): f for f in frames1}
        frames2_dict = {extract_frame_num(f): f for f in frames2}
        
        # Find common frame indices
        common_indices = sorted(set(frames1_dict.keys()) & set(frames2_dict.keys()))
        
        synchronized_pairs = [(frames1_dict[idx], frames2_dict[idx]) for idx in common_indices]
        
        return synchronized_pairs
