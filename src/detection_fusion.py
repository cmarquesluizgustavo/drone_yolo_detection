import math
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection from a single drone."""
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
    """Handles preparation of dual-drone measurements for geometric triangulation."""

    def __init__(self, association_threshold_m: float = 2.0, weapon_threshold: float = 0.5):
        self.association_threshold_m = association_threshold_m
        self.weapon_threshold = weapon_threshold
        
    def fuse_confidence(self, conf1, conf2):
        return 1.0 - (1.0 - conf1) * (1.0 - conf2)

    def prepare_measurements_for_triangulation(self, detections1, detections2, camera1, camera2):
        """Prepare paired measurements for geometric triangulation."""
        from geoconverter import GeoConverter
        
        # Get UAV positions in ground plane coordinates
        uav1_x, uav1_y = GeoConverter.geo_to_xy(camera1.lat, camera1.lon)
        uav2_x, uav2_y = GeoConverter.geo_to_xy(camera2.lat, camera2.lon)
        
        # Debug: print UAV positions
        if len(detections1) > 0 or len(detections2) > 0:
            import sys
            if hasattr(sys.stdout, 'isatty') and getattr(self, '_debug_counter', 0) == 0:
                print(f"    DEBUG Triangulation input:")
                print(f"      UAV1 xy=({uav1_x:.2f}, {uav1_y:.2f}) from geo=({camera1.lat:.6f}, {camera1.lon:.6f})")
                print(f"      UAV2 xy=({uav2_x:.2f}, {uav2_y:.2f}) from geo=({camera2.lat:.6f}, {camera2.lon:.6f})")
                if detections1:
                    det1 = detections1[0]
                    print(f"      D1: distance={det1.distance_m:.2f}m, bearing={det1.bearing_deg:.1f}°, det_xy=({det1.x:.2f}, {det1.y:.2f})")
                if detections2:
                    det2 = detections2[0]
                    print(f"      D2: distance={det2.distance_m:.2f}m, bearing={det2.bearing_deg:.1f}°, det_xy=({det2.x:.2f}, {det2.y:.2f})")
                self._debug_counter = 1
        
        measurement_groups = []
        used_det2 = set()
        
        # Process detections from drone 1
        for det1 in detections1:
            # Skip invalid detections
            if not (math.isfinite(det1.x) and math.isfinite(det1.y)):
                measurement_groups.append(self._create_single_measurement(
                    det1, (uav1_x, uav1_y), 'drone1'
                ))
                continue
            
            # Try to find corresponding detection from drone 2
            best_match_idx = None
            best_distance = self.association_threshold_m
            
            for idx2, det2 in enumerate(detections2):
                if idx2 in used_det2:
                    continue
                if not (math.isfinite(det2.x) and math.isfinite(det2.y)):
                    continue
                
                # Distance in ground plane
                dx = det1.x - det2.x
                dy = det1.y - det2.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < best_distance:
                    best_match_idx = idx2
                    best_distance = distance
            
            if best_match_idx is not None:
                # Found pair - create dual measurement
                det2 = detections2[best_match_idx]
                used_det2.add(best_match_idx)
                
                measurement_groups.append(self._create_dual_measurement(
                    det1, det2, (uav1_x, uav1_y), (uav2_x, uav2_y)
                ))
            else:
                # No match - single measurement
                measurement_groups.append(self._create_single_measurement(
                    det1, (uav1_x, uav1_y), 'drone1'
                ))
        
        # Add unmatched detections from drone 2
        for idx2, det2 in enumerate(detections2):
            if idx2 not in used_det2 and math.isfinite(det2.x) and math.isfinite(det2.y):
                measurement_groups.append(self._create_single_measurement(
                    det2, (uav2_x, uav2_y), 'drone2'
                ))
        
        return measurement_groups

    def _create_single_measurement(self, det, uav_pos, drone_name):
        return {
            'drone_measurements': [{
                'uav_pos': uav_pos,
                'distance': det.distance_m,
                'bearing': det.bearing_deg
            }],
            'person_confidence': det.person_confidence,
            'has_weapon': det.has_weapon,
            'weapon_confidence': det.weapon_confidence,
            f'bbox_{drone_name}': det.bbox,
            'drone_id': det.drone_id,
            'frame_id': det.frame_id
        }
    
    def _create_dual_measurement(self, det1, det2, uav1_pos, uav2_pos):
        """Create measurement group from paired drone detections."""
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
                    'bearing': det1.bearing_deg
                },
                {
                    'uav_pos': uav2_pos,
                    'distance': det2.distance_m,
                    'bearing': det2.bearing_deg
                }
            ],
            'person_confidence': fused_person_conf,
            'has_weapon': det1.has_weapon or det2.has_weapon,
            'weapon_confidence': fused_weapon_conf,
            'bbox_drone1': det1.bbox,
            'bbox_drone2': det2.bbox,
            'drone_ids': [det1.drone_id, det2.drone_id],
            'frame_id': det1.frame_id
        }
