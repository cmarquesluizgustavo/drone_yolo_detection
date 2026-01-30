import math
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class DroneState:
    """Represents the state of a drone at a given time."""
    gps_lat: float
    gps_lon: float
    altitude_m: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    
    
@dataclass
class Detection:
    """Represents a single detection from one drone."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    distance_m: float
    bearing_deg: float
    ground_x: float  # Ground-plane X coordinate (meters)
    ground_y: float  # Ground-plane Y coordinate (meters)
    gps_lat: float
    gps_lon: float
    drone_id: int
    frame_id: int
    has_weapon: bool = False
    weapon_confidence: float = 0.0


class DualDroneFusion:
    """Handles fusion of detections from two drones."""

    def __init__(self, association_threshold_m = 2.0):
        # maximum distance between detections to be considered the same target
        self.association_threshold_m = association_threshold_m
        
    def fuse_confidence(self, conf1, conf2) -> float:
        """
        Fuse confidence scores using probabilistic evidence accumulation.
        
        Under assumption of conditional independence:
        c_fused = 1 - (1 - c1)(1 - c2)
        """

        # Weighted version: c_f = 1 - (1-c1)^w1 * (1-c2)^w2
        return 1 - (1 - conf1) * (1 - conf2)

    def associate_detections(self, detections1, detections2):
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
                dx = det1.ground_x - det2.ground_x
                dy = det1.ground_y - det2.ground_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < self.association_threshold and distance < best_distance:
                    best_match = (idx2, det2)
                    best_distance = distance
            
            if best_match is not None:
                # Found a match - fuse the detections
                idx2, det2 = best_match
                used_det2.add(idx2)
                
                fused = self._fuse_detection_pair(det1, det2)
                fused_detections.append(fused)
            else:
                # No match - keep detection from drone 1 only
                fused_detections.append(self._detection_to_dict(det1))
        
        # Add unmatched detections from drone 2
        for idx2, det2 in enumerate(detections2):
            if idx2 not in used_det2:
                fused_detections.append(self._detection_to_dict(det2))
        
        return fused_detections
    
    def _fuse_detection_pair(self, det1: Detection, det2: Detection) -> Dict:
        """Fuse a pair of matched detections."""
        # Fuse confidence
        fused_conf = self.fuse_confidence(det1.confidence, det2.confidence)
        
        # Fuse weapon detection
        if det1.has_weapon or det2.has_weapon:
            weapon_conf1 = det1.weapon_confidence if det1.has_weapon else 0.0
            weapon_conf2 = det2.weapon_confidence if det2.has_weapon else 0.0
            fused_weapon_conf = self.fuse_confidence(weapon_conf1, weapon_conf2)
            has_weapon = True
        else:
            fused_weapon_conf = 0.0
            has_weapon = False
        
        # Fuse distance (simple average)
        fused_distance = self.fuse_distance_average(det1.distance_m, det2.distance_m)
        
        # Average ground position
        fused_x = (det1.ground_x + det2.ground_x) / 2.0
        fused_y = (det1.ground_y + det2.ground_y) / 2.0
        
        # Average GPS (simple approach)
        fused_lat = (det1.gps_lat + det2.gps_lat) / 2.0
        fused_lon = (det1.gps_lon + det2.gps_lon) / 2.0
        
        return {
            'confidence': fused_conf,
            'distance_m': fused_distance,
            'ground_x': fused_x,
            'ground_y': fused_y,
            'gps_lat': fused_lat,
            'gps_lon': fused_lon,
            'has_weapon': has_weapon,
            'weapon_confidence': fused_weapon_conf,
            'source': 'fused',
            'drone_ids': [det1.drone_id, det2.drone_id],
            'frame_id': det1.frame_id,
            'bbox_drone1': det1.bbox,
            'bbox_drone2': det2.bbox
        }
    
    def _detection_to_dict(self, det: Detection) -> Dict:
        """Convert a single Detection to dictionary format."""
        return {
            'confidence': det.confidence,
            'distance_m': det.distance_m,
            'ground_x': det.ground_x,
            'ground_y': det.ground_y,
            'gps_lat': det.gps_lat,
            'gps_lon': det.gps_lon,
            'has_weapon': det.has_weapon,
            'weapon_confidence': det.weapon_confidence,
            'source': f'drone{det.drone_id}',
            'drone_ids': [det.drone_id],
            'frame_id': det.frame_id,
            'bbox': det.bbox
        }


class FrameSynchronizer:
    """
    Synchronizes frames from two drone video streams.
    
    Handles temporal alignment based on timestamps or frame indices.
    """

    def __init__(self, max_time_diff_ms=100):
        # maximum time difference (milliseconds) to consider frames synchronized
        self.max_time_diff = max_time_diff_ms

    def synchronize_by_frame_index(self, frames1, frames2):
        """
        Synchronize frames by matching frame indices/numbers.
        
        Assumes frame filenames contain frame numbers (e.g., frame_0000.jpg).
        
        Args:
            frames1: List of frame paths from drone 1
            frames2: List of frame paths from drone 2
            
        Returns:
            List of tuples (frame1_path, frame2_path) for synchronized frames
        """
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
