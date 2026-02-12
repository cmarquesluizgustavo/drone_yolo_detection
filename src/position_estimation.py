import math
from geoconverter import GeoConverter


def estimate_distance(camera, pixel_height, real_height_m=1.7):
    if pixel_height <= 0:
        return 0.0
    return (real_height_m * camera.focal_length_px) / pixel_height


def estimate_distance_pitch(camera, y_pixel):
    delta_px = y_pixel - (camera.image_height_px / 2.0)
    pixel_angle_deg = (delta_px / (camera.image_height_px / 2.0)) * (camera.vertical_fov_deg / 2.0)
    alpha_deg = camera.pitch_deg + pixel_angle_deg
    if alpha_deg <= 0.0:
        return None
    tan_alpha = math.tan(math.radians(alpha_deg))
    return camera.height_m / tan_alpha


def estimate_bearing(camera, x_pixel):
    delta_px = x_pixel - (camera.image_width_px / 2.0)
    delta_angle = (delta_px / (camera.image_width_px / 2.0)) * (camera.horizontal_fov_deg / 2.0)
    bearing_deg = camera.yaw_deg + delta_angle
    return (bearing_deg + 360.0) % 360.0


def target_geoposition(camera, distance_m, bearing_deg):
    # Convert camera position to ground plane
    camera_x, camera_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    
    # Calculate target position in ground plane
    bearing_rad = math.radians(bearing_deg)
    x_target = camera_x + distance_m * math.sin(bearing_rad)
    y_target = camera_y + distance_m * math.cos(bearing_rad)
    
    # Convert back to geographic coordinates
    lat, lon = GeoConverter.xy_to_geo(x_target, y_target)
    return lat, lon


def bearing_from_geoposition(camera, target_lat, target_lon):
    camera_x, camera_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    target_x, target_y = GeoConverter.geo_to_xy(target_lat, target_lon)
    
    dx = target_x - camera_x
    dy = target_y - camera_y
    
    bearing_rad = math.atan2(dx, dy)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg


def distance_from_geoposition(camera, target_lat, target_lon):
    camera_x, camera_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    target_x, target_y = GeoConverter.geo_to_xy(target_lat, target_lon)
    
    return float(math.hypot(target_x - camera_x, target_y - camera_y))


# --- Fusion/triangulation methods for two UAVs ---
def fuse_average_target_positions_from_distance_bearing(camera1, dist1, bearing1, camera2, dist2, bearing2):
    # Compute target positions from each UAV
    x1, y1 = GeoConverter.geo_to_xy(camera1.lat, camera1.lon)
    x2, y2 = GeoConverter.geo_to_xy(camera2.lat, camera2.lon)
    b1_rad = math.radians(bearing1)
    b2_rad = math.radians(bearing2)
    xt1 = x1 + dist1 * math.sin(b1_rad)
    yt1 = y1 + dist1 * math.cos(b1_rad)
    xt2 = x2 + dist2 * math.sin(b2_rad)
    yt2 = y2 + dist2 * math.cos(b2_rad)
    # Average the two positions
    xt = (xt1 + xt2) / 2.0
    yt = (yt1 + yt2) / 2.0
    # Convert back to lat/lon
    lat, lon = GeoConverter.xy_to_geo(xt, yt)
    return lat, lon


def triangulate_target_by_bearing_intersection(camera1, bearing1, camera2, bearing2):
    x1, y1 = GeoConverter.geo_to_xy(camera1.lat, camera1.lon)
    x2, y2 = GeoConverter.geo_to_xy(camera2.lat, camera2.lon)
    theta1 = math.radians(bearing1)
    theta2 = math.radians(bearing2)
    # Line 1: (x1, y1) + t1 * (sin(theta1), cos(theta1))
    # Line 2: (x2, y2) + t2 * (sin(theta2), cos(theta2))
    # Solve for intersection:
    # x1 + t1*sin(theta1) = x2 + t2*sin(theta2)
    # y1 + t1*cos(theta1) = y2 + t2*cos(theta2)
    # Rearranged as a linear system:
    # [sin1, -sin2] [t1] = [x2 - x1]
    # [cos1, -cos2] [t2]   [y2 - y1]
    sin1, cos1 = math.sin(theta1), math.cos(theta1)
    sin2, cos2 = math.sin(theta2), math.cos(theta2)
    det = sin1 * cos2 - sin2 * cos1
    if abs(det) < 1e-8:
        return None  # Lines are parallel or nearly so
    dx = x2 - x1
    dy = y2 - y1
    t1 = (dx * cos2 - dy * sin2) / det
    # Intersection point
    xi = x1 + t1 * sin1
    yi = y1 + t1 * cos1
    lat, lon = GeoConverter.xy_to_geo(xi, yi)
    return lat, lon