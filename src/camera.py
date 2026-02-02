import math
import json
import os

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

class Camera:
    def __init__(self, sensor_width_mm=6.4, sensor_height_mm=4.8,
                 focal_35mm_mm=25.6, image_width_px=1920, image_height_px=1080,
                 lat=0.0, lon=0.0, camera_height_m=0.0,
                 yaw_deg=0.0, pitch_deg=0.0):
        
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px

        self.focal_35mm_mm = focal_35mm_mm

        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        
        # Drone state management
        self.lat = lat
        self.lon = lon
        self.camera_height_m = camera_height_m
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg
        
        # Metadata storage for logging
        self.timestamp_sec = None
        self.datetime = None
        
        # Calculate pixel sizes first
        self.pixel_size_x_mm = sensor_width_mm / image_width_px
        self.pixel_size_y_mm = sensor_height_mm / image_height_px
        
        # Calculate actual focal length from 35mm equivalent
        self.diag_sensor_mm = math.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
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


    def update_state(self, lat=None, lon=None, camera_height_m=None,
                    yaw_deg=None, pitch_deg=None):
        """Update drone state parameters."""
        if lat is not None:
            self.lat = lat
        if lon is not None:
            self.lon = lon
        if camera_height_m is not None:
            self.camera_height_m = camera_height_m
        if yaw_deg is not None:
            self.yaw_deg = yaw_deg
        if pitch_deg is not None:
            self.pitch_deg = pitch_deg

    def load_from_json(self, image_path):
        json_path = os.path.splitext(image_path)[0] + '.json'

        if not os.path.exists(json_path):
            return False

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            def _wrap_angle_deg(angle_deg: float) -> float:
                """Wrap an angle to [0, 360)."""
                return float((angle_deg % 360.0 + 360.0) % 360.0)

            def _wrap_signed_angle_deg(angle_deg: float) -> float:
                """Wrap an angle to [-180, 180)."""
                wrapped = _wrap_angle_deg(angle_deg)
                if wrapped >= 180.0:
                    wrapped -= 360.0
                return float(wrapped)

            # GPS
            self.lat = data['gps']['S']
            self.lon = data['gps']['W']
            # For ground-range geometry we only need a positive camera height.
            self.camera_height_m = abs(float(data['gps']['height_m']))

            # Orientation (ground gyro)
            # Normalize yaw to [0, 360) and pitch to a signed convention:
            #   pitch_deg > 0  => camera points DOWN by that many degrees
            #   pitch_deg = 0  => horizon
            #   pitch_deg < 0  => camera points UP
            yaw_raw = float(data['gpry']['yaw'])
            pitch_raw = float(data['gpry']['pitch'])
            self.yaw_deg = _wrap_angle_deg(yaw_raw)
            pitch_signed = _wrap_signed_angle_deg(pitch_raw)
            self.pitch_deg = float(-pitch_signed)

            # Metadata
            self.datetime = data['datetime']
            self.iso = data['iso']
            self.shutter = data['shutter']
            self.fnum = data['fnum']
            self.ev = data['ev']

            return True

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading metadata from {json_path}: {e}")
            return False

    
    def has_metadata(self, image_path):
        """Check if metadata JSON file exists for given image."""
        json_path = os.path.splitext(image_path)[0] + '.json'
        return os.path.exists(json_path)

    def get_state_dict(self):
        """Return current drone state as dictionary."""
        return {
            'lat': self.lat,
            'lon': self.lon,
            'camera_height_real_m': self.camera_height_m,
            'yaw_deg': self.yaw_deg,
            'pitch_deg': self.pitch_deg
        }

    def calculate_person_geoposition(self, camera_lat, camera_lon, camera_yaw_deg, x_pixel, distance_m):
        """Estimate target lat/lon from camera pose + pixel x + range.

        This is used to keep outputs consistent with the dual-drone pipeline
        (which computes bearing + range and then projects onto the ground plane).
        """
        if distance_m is None:
            return None, None

        # Ensure camera state matches the inputs used for this calculation.
        self.lat = camera_lat
        self.lon = camera_lon
        self.yaw_deg = camera_yaw_deg

        from position_estimation import estimate_bearing, simple_target_geoposition

        bearing_deg = estimate_bearing(self, x_pixel)
        lat, lon = simple_target_geoposition(self, float(distance_m), float(bearing_deg))
        return lat, lon