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
    cam_x, cam_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    
    # Calculate target position in ground plane
    bearing_rad = math.radians(bearing_deg)
    x_target = cam_x + distance_m * math.sin(bearing_rad)
    y_target = cam_y + distance_m * math.cos(bearing_rad)
    
    # Convert back to geographic coordinates
    lat, lon = GeoConverter.xy_to_geo(x_target, y_target)
    return lat, lon


def bearing_from_geoposition(camera, target_lat: float, target_lon: float) -> float:
    camera_x, camera_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    target_x, target_y = GeoConverter.geo_to_xy(target_lat, target_lon)
    
    dx = target_x - camera_x
    dy = target_y - camera_y
    
    bearing_rad = math.atan2(dx, dy)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg


def distance_from_geoposition(camera, target_lat: float, target_lon: float) -> float:
    camera_x, camera_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    target_x, target_y = GeoConverter.geo_to_xy(target_lat, target_lon)
    
    return float(math.hypot(target_x - camera_x, target_y - camera_y))