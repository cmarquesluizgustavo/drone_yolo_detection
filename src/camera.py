import math
import re
import os
from pathlib import PurePath, Path

"""
Camera model for Autel Robotics EVO 2 Dual V2

Specs:
- Image Sensor: 1/2" CMOS (6.4mm x 4.8mm)
- Pixels: 48MP (Still), Multiple video resolutions available
- Perspective (HFOV): 79°
- Lens EFL: 25.6 mm
- Aperture: f/1.8
- Video Resolution: 8K/6K/4K/2.7K/1080P at various framerates
"""

class Camera:
    def __init__(self, sensor_width_mm=6.4, sensor_height_mm=4.8, focal_35mm_mm=25.6, 
                 image_width_px=1920, image_height_px=1080 , fov_deg=79.0):
        
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px

        self.cx_px = (self.image_width_px - 1) / 2.0
        self.cy_px = (self.image_height_px - 1) / 2.0
        
        self.height_m = 0.0      # filename
        self.pitch_deg = 0.0     # telemetry/manual
        self.yaw_deg = 0.0       # telemetry/manual
        self.roll_deg = 0.0      # telemetry/manual
        self.lat = 0.0
        self.lon = 0.0
        
        self.hfov_deg = fov_deg

        hfov_rad = math.radians(fov_deg)
        self.fx_px = (self.image_width_px / 2.0) / math.tan(hfov_rad / 2.0)
        
        vfov_rad = 2.0 * math.atan(
            math.tan(hfov_rad / 2.0) * (self.image_height_px / self.image_width_px)
        )
        self.fy_px = (self.image_height_px / 2.0) / math.tan(vfov_rad / 2.0)

        self.vfov_deg = math.degrees(vfov_rad)

        # # ANOTHER WAY

        # self.focal_35mm_mm = focal_35mm_mm
        # self.sensor_width_mm = sensor_width_mm
        # self.sensor_height_mm = sensor_height_mm

        # #Calculate pixel sizes first
        # self.pixel_size_x_mm = sensor_width_mm / image_width_px
        # self.pixel_size_y_mm = sensor_height_mm / image_height_px
        
        # # Calculate actual focal length from 35mm equivalent
        # self.diag_sensor_mm = math.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
        # self.diag_35mm_mm = 43.27  # diagonal of 35mm sensor
        # self.crop_factor = self.diag_35mm_mm / self.diag_sensor_mm
        # self.focal_length_mm = self.focal_35mm_mm / self.crop_factor
        
        # self.focal_length_px = self.focal_length_mm / self.pixel_size_y_mm

        # #Focal length in pixels (separate axes!)
        # self.fx_px = self.focal_length_mm / self.pixel_size_x_mm
        # self.fy_px = self.focal_length_mm / self.pixel_size_y_mm

        # #Compute FOV from real optics
        # self.hfov_deg = math.degrees(
        #     2 * math.atan(sensor_width_mm / (2 * self.focal_length_mm))
        # )
        # self.vfov_deg = math.degrees(
        #     2 * math.atan(sensor_height_mm / (2 * self.focal_length_mm))
        # )
        
    
    def load_telemetry_from_video_path(self, video_or_image_path):
        file_path = video_or_image_path
        
        # If it's already a txt file, use it directly
        if file_path.endswith('.txt'):
            txt_path = file_path
        else:
            # Extract pattern from filename: (real|falso)_<distance>_<height>
            basename = os.path.basename(file_path)
            pattern = r'(real|falso)_(\d+)_(\d+)'
            match = re.search(pattern, basename, re.IGNORECASE)
            if not match:
                # Frame paths usually look like: .../<sample_name>/frames/frame_0001.jpg
                # so the metadata is in the parent folder name, not the frame filename.
                match = re.search(pattern, str(file_path), re.IGNORECASE)
            if not match:
                return False
            
            # Find drone and angle from path
            path_parts = Path(file_path).parts
            drone_id = next((p for p in path_parts if p.startswith('drone')), None)
            angle = path_parts[path_parts.index(drone_id) + 1] if drone_id and path_parts.index(drone_id) + 1 < len(path_parts) else None
            
            if not drone_id or not angle:
                return False
            
            # Build txt path: inputs/raw/{drone_id}/{angle}/{base_name}.txt
            root_path = Path(*path_parts[:path_parts.index('inputs')]) if 'inputs' in path_parts else Path.cwd()
            video_base_name = f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
            txt_path = root_path / 'inputs' / 'raw' / drone_id / angle / f"{video_base_name}.txt"

            # Also set camera height from the dataset naming convention.
            try:
                self.height_m = float(match.group(3))
            except ValueError:
                pass
        
        # Load telemetry from txt file
        if not os.path.exists(txt_path):
            return False
        
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    if ':' not in line:
                        continue
                    key, value = line.strip().split(':', 1)
                    key, value = key.strip(), value.strip()
                    try:
                        if key == 'pitch':
                            self.pitch_deg = -float(value)
                        elif key == 'roll':
                            self.roll_deg = float(value)
                        elif key == 'yaw':
                            self.yaw_deg = float(value)
                        elif key == 'lat':
                            self.lat = float(value)
                        elif key == 'lon':
                            self.lon = float(value)
                    except ValueError:
                        continue

            # Debug print for loaded telemetry
            # print(f"[Camera Telemetry] Loaded for {txt_path}: lat={self.lat}, lon={self.lon}, yaw={self.yaw_deg}, pitch={self.pitch_deg}, roll={self.roll_deg}")
            # if self.lat == 0.0 or self.lon == 0.0:
            #     print(f"[Camera Telemetry WARNING] lat/lon is zero for {txt_path}")
            return True
        except Exception as e:
            print(f"Error loading telemetry from {txt_path}: {e}")
            return False