import math
from geoconverter import GeoConverter


def estimate_distance(camera, pixel_height, real_height_m=1.7):
    if pixel_height is None or pixel_height <= 0:
        return None
    return (real_height_m * camera.fy_px) / pixel_height

def estimate_distance_pitch(camera, y_pixel):
    if y_pixel is None:
        return None
    
    #delta_px = y_pixel - (camera.image_height_px / 2.0)
    #delta_angle_rad = math.radians((delta_px / (camera.image_height_px / 2.0)) * (camera.vfov_deg / 2.0))
    
    delta_angle_rad = math.atan((y_pixel - camera.cy_px) / camera.fy_px)
    alpha_rad = math.radians(camera.pitch_deg) + delta_angle_rad

    tan_alpha = math.tan(alpha_rad)
    if abs(tan_alpha) <= 1e-6:
        return None

    return camera.height_m / tan_alpha

def estimate_bearing(camera, x_pixel):
    # delta_px = x_pixel - (camera.image_width_px / 2.0)
    # delta_angle = (delta_px / (camera.image_width_px / 2.0)) * (camera.hfov_deg / 2.0)
    delta_angle = math.degrees(math.atan((x_pixel - camera.cx_px) / camera.fx_px))
    
    bearing_deg = camera.yaw_deg + delta_angle
    return (bearing_deg + 360.0) % 360.0

def target_geoposition(camera, distance_m, bearing_deg):
    return GeoConverter.polar_to_geo(camera.lat, camera.lon, bearing_deg, distance_m)

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

def _target_xy(camera, dist, bearing):
    x, y = GeoConverter.geo_to_xy(camera.lat, camera.lon)
    b = math.radians(bearing)
    return (x + dist * math.sin(b), y + dist * math.cos(b))

def fuse_avg_position_from_distance_bearing(camera1, dist1, bearing1, camera2, dist2, bearing2):
    p1 = _target_xy(camera1, dist1, bearing1)
    p2 = _target_xy(camera2, dist2, bearing2)

    w1, w2 = weight_function(1,1)

    x = (w1*p1[0] + w2*p2[0]) / (w1 + w2)
    y = (w1*p1[1] + w2*p2[1]) / (w1 + w2)

    return GeoConverter.xy_to_geo(x, y)

def weight_function(a,b):
    #INSER HERE THE WEIGHT FUNCTION
    return a, b

def fuse_triangulate_position_from_bearing_intersection(camera1, bearing1, camera2, bearing2):
    x1, y1 = GeoConverter.geo_to_xy(camera1.lat, camera1.lon)
    x2, y2 = GeoConverter.geo_to_xy(camera2.lat, camera2.lon)

    t1 = math.radians(bearing1)
    t2 = math.radians(bearing2)

    # direction vectors
    d1x, d1y = math.sin(t1), math.cos(t1)
    d2x, d2y = math.sin(t2), math.cos(t2)

    # normals
    n1x, n1y = -d1y, d1x
    n2x, n2y = -d2y, d2x

    # b = n·p0
    b1 = n1x*x1 + n1y*y1
    b2 = n2x*x2 + n2y*y2

    # A^T A (2x2)
    a11 = n1x*n1x + n2x*n2x
    a12 = n1x*n1y + n2x*n2y
    a22 = n1y*n1y + n2y*n2y

    det = a11*a22 - a12*a12

    # A^T b
    c1 = n1x*b1 + n2x*b2
    c2 = n1y*b1 + n2y*b2

    px = ( a22*c1 - a12*c2) / det
    py = (-a12*c1 + a11*c2) / det

    return GeoConverter.xy_to_geo(px, py)

# def weights_from_detection(person_conf, dist, bbox):
#     x1, y1, x2, y2 = bbox
#     bw = max(1.0, x2-x1)
#     bh = max(1.0, y2-y1)
#     size = (bw*bh) ** 0.5
#     w_avg   = person_conf * (size**2) / (dist**2)      # for point avg
#     return w_avg

# def fuse_avg_position_from_distance_bearing_weighted(camera1, dist1, bearing1, bbox1, conf1, camera2, dist2, bearing2, bbox2, conf2):
#     p1 = _target_xy(camera1, dist1, bearing1)
#     p2 = _target_xy(camera2, dist2, bearing2)
    
#     w1 = weights_from_detection(conf1, dist1, bbox1)
#     w2 = weights_from_detection(conf2, dist2, bbox2)
    
#     total = w1 + w2
    
#     if total <= 1e-6:
#         # fallback to simple average
#         xt = (p1[0] + p2[0]) / 2.0
#         yt = (p1[1] + p2[1]) / 2.0
#     else:
#         xt = (w1*p1[0] + w2*p2[0]) / total
#         yt = (w1*p1[1] + w2*p2[1]) / total
    
#     return GeoConverter.xy_to_geo(xt, yt)