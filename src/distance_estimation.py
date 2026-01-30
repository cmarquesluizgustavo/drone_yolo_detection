import math
import numpy as np
from geoconverter import GeoConverter
from scipy.optimize import least_squares

class Camera:
    """
    Camera model for Autel Robotics EVO 2 Dual V2
    
    Specs:
    - Image Sensor: 1/2" CMOS (6.4mm x 4.8mm)
    - Pixels: 48MP (Still), Multiple video resolutions available
    - Perspective (HFOV): 79°
    - Lens EFL: 25.6 mm
    - Aperture: f/2.8–f/11
    - Video Resolution: 8K/6K/4K/2.7K/1080P at various framerates
    """
    def __init__(self, sensor_width_mm=6.4, sensor_height_mm=4.8,
                 focal_35mm_mm=25.6, image_width_px=1920, image_height_px=1080):
        
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px

        self.focal_35mm_mm = focal_35mm_mm

        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        
        # Calculate pixel sizes first
        self.pixel_size_x_mm = sensor_width_mm / image_width_px
        self.pixel_size_y_mm = sensor_height_mm / image_height_px
        
        # Calculate actual focal length from 35mm equivalent
        self.diag_sensor_mm = np.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
        self.diag_35mm_mm = 43.27  # diagonal of 35mm sensor
        self.crop_factor = self.diag_35mm_mm / self.diag_sensor_mm
        self.focal_length_mm = self.focal_35mm_mm / self.crop_factor
        
        self.focal_length_px = self.focal_length_mm / self.pixel_size_y_mm

        # Compute FOV from real optics
        self.horizontal_fov_deg = math.degrees(
            2 * math.atan(sensor_width_mm / (2 * self.focal_length_mm))
        )
        self.vertical_fov_deg = math.degrees(
            2 * math.atan(sensor_height_mm / (2 * self.focal_length_mm))
        )
        # Note: Manufacturer spec lists perspective as 79° (horizontal FOV)

    def estimate_distance(self, pixel_height, real_height_m=1.7):
        return (real_height_m * self.focal_length_px) / pixel_height
    
    def estimate_distance_2(self, y_pixel, camera_tilt_deg, camera_height_m):
        
        pixel_offset = (y_pixel - self.image_height_px / 2) / self.image_height_px
        pixel_angle_deg = pixel_offset * self.vertical_fov_deg

        alpha_deg = camera_tilt_deg + pixel_angle_deg

        if alpha_deg <= 0:
            return None

        return camera_height_m / math.tan(math.radians(alpha_deg))

    def calculate_bearing(camera_yaw_deg, x_pixel, image_width):

        delta_px = x_pixel - (image_width / 2.0)

        delta_angle = (delta_px / (image_width / 2.0)) * (self.horizontal_fov_deg / 2.0)

        bearing_deg = camera_yaw_deg + delta_angle

        return (bearing_deg + 360.0) % 360.0

    def extract_geoposition(camera_lat, camera_lon, bearing_deg, distance_m):

        return GeoConverter.polar_to_geo(
            camera_lat,
            canera_lon,
            bearing_deg,
            distance_m
        )

    def extract_geoposition_dual_drone(
        lat1, lon1, distance1_m, bearing1_deg,
        lat2, lon2, distance2_m, bearing2_deg
    ):

        b1 = math.radians(bearing1_deg)
        b2 = math.radians(bearing2_deg)

        x1, y1 = Converter.geo_to_xy(lat1, lon1)
        x2, y2 = Converter.geo_to_xy(lat2, lon2)

        def residuals(p):
            x, y = p

            distance_res_1 = math.hypot(x - x1, y - y1) - distance1_m
            distance_res_2 = math.hypot(x - x2, y - y2) - distance2_m

       
            bearing_res_1 = math.atan2(x - x1, y - y1) - b1
            bearing_res_2 = math.atan2(x - x2, y - y2) - b2

            return [
                distance_res_1,
                bearing_res_1,
                distance_res_2,
                bearing_res_2
            ]

        x0 = (x1 + x2) / 2.0
        y0 = (y1 + y2) / 2.0

        result = least_squares(residuals, [x0, y0])

        lat, lon = Converter.xy_to_geo(result.x[0], result.x[1])
        return lat, lon

    def estimations_after_fusion(drone_pos, target_pos):

        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]

        distance_m = float(np.hypot(dx, dy))

        bearing_rad = math.atan2(dx, dy)
        bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0

        return distance_m, bearing_deg