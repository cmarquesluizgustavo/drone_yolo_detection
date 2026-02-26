import math
from geoconverter import GeoConverter


def estimate_distance(camera, pixel_height, real_height_m=1.7):
    return (real_height_m * camera.focal_length_px) / pixel_height

def estimate_distance_pitch(camera, y_pixel):
    #delta_px = y_pixel - (camera.image_height_px / 2.0)
    #delta_angle = (delta_px / (camera.image_height_px / 2.0)) * (camera.vertical_fov_deg / 2.0)
    
    cy = camera.image_height_px / 2.0
    f = camera.focal_length_px
    delta_angle = math.degrees(math.atan((y_pixel - cy) / f))
    alpha_deg = camera.pitch_deg + delta_angle
    
    tan_alpha = math.tan(math.radians(alpha_deg))
    return camera.height_m / tan_alpha

# def estimate_distance_fused(camera, pixel_height, y_pixel):
#     d_h = estimate_distance(camera, pixel_height)
#     d_g = estimate_distance_pitch(camera, y_pixel)

#     # alpha (radians)
#     cy = camera.image_height_px / 2.0
#     f  = camera.focal_length_px
#     alpha = math.radians(camera.pitch_deg) + math.atan((y_pixel - cy) / f)

#     # If alpha is too small or negative, ground-plane is unreliable
#     if alpha < math.radians(3.0):
#         return d_h

#     sigma_h_rel = 0.30
#     sigma_h = sigma_h_rel * d_h

#     sigma_alpha = math.radians(1.5)   # combined pitch + pixel noise (rad)
#     sigma_alt   = 0.30               # altitude noise (m)

#     h = camera.height_m
#     sin_a = math.sin(alpha)
#     tan_a = math.tan(alpha)

#     # avoid blow-ups
#     sin_a = max(sin_a, 1e-3)
#     tan_a = max(tan_a, 1e-3)

#     # var(d) ≈ (h/sin^2(a))^2 * var(a) + (1/tan(a))^2 * var(h)
#     sigma_g = math.sqrt(((h / (sin_a**2))**2) * (sigma_alpha**2) +
#                         ((1.0 / tan_a)**2) * (sigma_alt**2))

#     # inverse-variance fusion
#     w_h = 1.0 / (sigma_h * sigma_h)
#     w_g = 1.0 / (sigma_g * sigma_g)

#     return (w_h * d_h + w_g * d_g) / (w_h + w_g)

def estimate_distance_fused(camera, pixel_height, y_pixel):
    d_h = estimate_distance(camera, pixel_height)
    d_g = estimate_distance_pitch(camera, y_pixel)
    return (d_h + d_g) / 2.0

def estimate_bearing(camera, x_pixel):
    #delta_px = x_pixel - (camera.image_width_px / 2.0)
    #delta_angle = (delta_px / (camera.image_width_px / 2.0)) * (camera.horizontal_fov_deg / 2.0)
    
    cx = camera.image_width_px / 2.0
    f = camera.focal_length_px
    delta_angle = math.degrees(math.atan((x_pixel - cx) / f))
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
    xt = (p1[0] + p2[0]) / 2.0
    yt = (p1[1] + p2[1]) / 2.0
    return GeoConverter.xy_to_geo(xt, yt)

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

def weights_from_detection(person_conf, dist, bbox):
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2-x1)
    bh = max(1.0, y2-y1)
    size = (bw*bh) ** 0.5

    # w_avg   = person_conf
    # w_avg   = person_conf * size
    #w_avg   = person_conf / dist
    w_avg   = person_conf * size / dist      # for point avg
    
    #w_bi = person_conf 
    #w_bi = person_conf * size 
    #w_bi = person_conf / dist
    #w_bi = person_conf * size / dist         # for bearing triangulation

    return w_avg#, w_bi

# def weights_from_detection(person_conf, dist, bbox):

#     x1,y1,x2,y2 = bbox

#     bw = max(1.0, x2-x1)
#     bh = max(1.0, y2-y1)

#     size = (bw*bh) ** 0.5

#     c = max(0.001, min(1.0, person_conf))
#     #d = 0.0 if dist is None else float(dist)
#     #d = max(1e-3, d)

#     #sigma_d = d / (conf * size)
#     #sigma_theta = d / (conf * size)

#     # --- tuned for your bbox sizes ---
#     sigma_d0 = 0.25
#     k_size_d = 40
#     k_conf_d = 1.5

#     sigma_theta0 = 1.0
#     k_size_theta = 80
#     #k_conf_theta = 5

#     sigma_d = sigma_d0 + k_size_d/size + k_conf_d*(1-c)
#     #sigma_theta = sigma_theta0 + k_size_theta/size + k_conf_theta*(1-c)

#     w_avg = 1.0 / (sigma_d**2)
#     #w_bi = 1.0 / (sigma_theta**2)

#     return w_avg#, w_bi

def fuse_avg_position_from_distance_bearing_weighted(camera1, dist1, bearing1, bbox1, conf1, camera2, dist2, bearing2, bbox2, conf2):
    p1 = _target_xy(camera1, dist1, bearing1)
    p2 = _target_xy(camera2, dist2, bearing2)
    
    w1 = weights_from_detection(conf1, dist1, bbox1)
    w2 = weights_from_detection(conf2, dist2, bbox2)
    
    total = w1 + w2
    
    if total <= 1e-6:
        # fallback to simple average
        xt = (p1[0] + p2[0]) / 2.0
        yt = (p1[1] + p2[1]) / 2.0
    else:
        xt = (w1*p1[0] + w2*p2[0]) / total
        yt = (w1*p1[1] + w2*p2[1]) / total
    
    return GeoConverter.xy_to_geo(xt, yt)

# def fuse_triangulate_position_from_bearing_intersection_weighted(camera1, dist1, bearing1, bbox1, conf1, camera2, dist2, bearing2, bbox2, conf2):
#     x1, y1 = GeoConverter.geo_to_xy(camera1.lat, camera1.lon)
#     x2, y2 = GeoConverter.geo_to_xy(camera2.lat, camera2.lon)

#     t1 = math.radians(bearing1)
#     t2 = math.radians(bearing2)

#     # Direction vectors
#     d1x, d1y = math.sin(t1), math.cos(t1)
#     d2x, d2y = math.sin(t2), math.cos(t2)

#     # Line normals
#     n1x, n1y = -d1y, d1x
#     n2x, n2y = -d2y, d2x

#     # Each line: n·p = n·p0
#     b1 = n1x * x1 + n1y * y1
#     b2 = n2x * x2 + n2y * y2
    
#     _, w1 = weights_from_detection(conf1, dist1, bbox1)
#     _, w2 = weights_from_detection(conf2, dist2, bbox2)

#     # Weighted normal equations for 2x2:
#     a11 = w1*n1x*n1x + w2*n2x*n2x
#     a12 = w1*n1x*n1y + w2*n2x*n2y
#     a22 = w1*n1y*n1y + w2*n2y*n2y

#     det = a11*a22 - a12*a12

#     c1 = w1*n1x*b1 + w2*n2x*b2
#     c2 = w1*n1y*b1 + w2*n2y*b2

#     px = ( a22*c1 - a12*c2) / det
#     py = (-a12*c1 + a11*c2) / det

#     return GeoConverter.xy_to_geo(px, py)