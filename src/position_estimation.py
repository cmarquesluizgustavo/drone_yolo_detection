import math
from geoconverter import GeoConverter
from scipy.optimize import least_squares

def estimate_distance(camera, pixel_height, real_height_m=1.7):
    return (real_height_m * camera.focal_length_px) / pixel_height

def estimate_distance_2(camera, y_pixel):
    delta_px = y_pixel - (camera.image_height_px / 2.0)
    pixel_angle_deg = (delta_px / (camera.image_height_px / 2.0)) * (camera.vertical_fov_deg / 2.0)
    alpha_deg = camera.pitch_deg + pixel_angle_deg

    # Guard against invalid/unstable geometry: near horizon (tan ~ 0) or above horizon (alpha <= 0)
    if alpha_deg <= 0.0:
        return None
    tan_alpha = math.tan(math.radians(alpha_deg))
    if abs(tan_alpha) < 1e-6:
        return None
    return camera.height_m / tan_alpha

def estimate_bearing(camera, x_pixel):
    delta_px = x_pixel - (camera.image_width_px / 2.0)
    delta_angle = (delta_px / (camera.image_width_px / 2.0)) * (camera.horizontal_fov_deg / 2.0)
    bearing_deg = camera.yaw_deg + delta_angle
    return (bearing_deg + 360.0) % 360.0

def simple_target_geoposition(camera, distance_m, bearing_deg):
    cam_x, cam_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    bearing_rad = math.radians(bearing_deg)
    x_target = cam_x + distance_m * math.sin(bearing_rad)
    y_target = cam_y + distance_m * math.cos(bearing_rad)
    lat, lon = GeoConverter.xy_to_geo(x_target, y_target)
    return lat, lon

def fuse_target_geoposition(camera1, distance1_m, bearing1_deg, camera2, distance2_m, bearing2_deg):
    b1 = math.radians(bearing1_deg)
    b2 = math.radians(bearing2_deg)

    x1, y1 = GeoConverter.geo_to_xy(camera1.lat, camera1.lon)
    x2, y2 = GeoConverter.geo_to_xy(camera2.lat, camera2.lon)

    def _wrap_angle_rad(a: float) -> float:
        return float((a + math.pi) % (2.0 * math.pi) - math.pi)

    def residuals(p):
        x, y = p
        distance_res_1 = math.hypot(x - x1, y - y1) - distance1_m
        distance_res_2 = math.hypot(x - x2, y - y2) - distance2_m
        bearing_res_1 = _wrap_angle_rad(math.atan2(x - x1, y - y1) - b1)
        bearing_res_2 = _wrap_angle_rad(math.atan2(x - x2, y - y2) - b2)
        return [distance_res_1, bearing_res_1, distance_res_2, bearing_res_2]

    x0 = (x1 + x2) / 2.0
    y0 = (y1 + y2) / 2.0
    result = least_squares(residuals, [x0, y0])
    lat, lon = GeoConverter.xy_to_geo(result.x[0], result.x[1])
    return lat, lon

def bearing_from_position(camera, target_lat, target_lon):
    camera_x, camera_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    target_x, target_y = GeoConverter.geo_to_xy(target_lat, target_lon)
    dx = target_x - camera_x
    dy = target_y - camera_y
    bearing_rad = math.atan2(dx, dy)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg

def distance_from_position(camera, target_lat, target_lon):
    camera_x, camera_y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    target_x, target_y = GeoConverter.geo_to_xy(target_lat, target_lon)
    return float(math.hypot(target_x - camera_x, target_y - camera_y))