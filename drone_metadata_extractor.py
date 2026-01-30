"""
Drone Metadata Extractor
Extracts and parses telemetry data from drone video subtitle tracks.

The metadata includes:
- HOME position (takeoff location)
- GPS coordinates (current position and altitude)
- Camera settings (ISO, Shutter, EV, F-Number)
- F.PRY: Frame Pitch, Roll, Yaw (camera gimbal orientation)
- G.PRY: GPS/Gyro Pitch, Roll, Yaw (drone body orientation)
"""

import subprocess
import re
from dataclasses import dataclass

@dataclass
class DroneMetadata:
    """Represents drone telemetry data at a specific timestamp."""
    timestamp: float  # seconds from start
    datetime_str: str
    
    # HOME position (takeoff location)
    home_longitude: float  # W coordinate
    home_latitude: float   # S coordinate
    
    # Current GPS position
    gps_longitude: float
    gps_latitude: float
    gps_altitude: float  # meters
    
    # Camera settings
    iso: int
    shutter_speed: int
    exposure_value: float
    f_number: float
    
    # Frame PRY (camera gimbal orientation)
    frame_pitch: float
    frame_roll: float
    frame_yaw: float
    
    # GPS/Gyro PRY (drone body orientation)
    gyro_pitch: float
    gyro_roll: float
    gyro_yaw: float


class DroneMetadataExtractor:
    """Extracts telemetry data from drone video subtitle tracks."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self._subtitle_data = None
    
    def extract_subtitle_text(self):
        """
        Extract the subtitle track from the video using ffmpeg.
        
        Returns:
            Raw subtitle text in SRT format
        """
        if self._subtitle_data is not None:
            return self._subtitle_data
        
        try:
            cmd = [
                'ffmpeg',
                '-i', self.video_path,
                '-map', '0:s:0',  # Map first subtitle stream
                '-c:s', 'text',
                '-f', 'srt',
                '-'
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
            if result.returncode == 0:
                self._subtitle_data = result.stdout
                return self._subtitle_data
            else:
                raise RuntimeError(f"ffmpeg failed to extract subtitles: {result.stderr}")
        
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
    
    def parse_metadata_entry(self, entry_text, timestamp):
        """
        Parse a single subtitle entry into structured metadata.
        
        Args:
            entry_text: The text content of one subtitle entry
            timestamp: The timestamp in seconds
            
        Returns:
            DroneMetadata object or None if parsing fails
        """
        try:
            # Parse HOME position
            home_match = re.search(r'HOME\(W:\s*([\d.]+),\s*S:\s*([\d.]+)\)\s*([\d-]+\s+[\d:]+)', entry_text)
            if not home_match:
                return None
            
            home_lon = float(home_match.group(1))
            home_lat = float(home_match.group(2))
            datetime_str = home_match.group(3)
            
            # Parse GPS position
            gps_match = re.search(r'GPS\(W:\s*([\d.]+),\s*S:\s*([\d.]+),\s*([\d.]+)m\)', entry_text)
            if not gps_match:
                return None
            
            gps_lon = float(gps_match.group(1))
            gps_lat = float(gps_match.group(2))
            gps_alt = float(gps_match.group(3))
            
            # Parse camera settings
            iso_match = re.search(r'ISO:(\d+)', entry_text)
            shutter_match = re.search(r'SHUTTER:(\d+)', entry_text)
            ev_match = re.search(r'EV:([-\d.]+)', entry_text)
            fnum_match = re.search(r'F-NUM:([\d.]+)', entry_text)
            
            if not all([iso_match, shutter_match, ev_match, fnum_match]):
                return None
            
            iso = int(iso_match.group(1))
            shutter = int(shutter_match.group(1))
            ev = float(ev_match.group(1))
            f_num = float(fnum_match.group(1))
            
            # Parse F.PRY (Frame/Camera orientation)
            fpry_match = re.search(r'F\.PRY\s*\(([-\d.]+)°,\s*([-\d.]+)°,\s*([-\d.]+)°\)', entry_text)
            if not fpry_match:
                return None
            
            frame_pitch = float(fpry_match.group(1))
            frame_roll = float(fpry_match.group(2))
            frame_yaw = float(fpry_match.group(3))
            
            # Parse G.PRY (Gyro/Drone orientation)
            gpry_match = re.search(r'G\.PRY\s*\(([-\d.]+)°,\s*([-\d.]+)°,\s*([-\d.]+)°\)', entry_text)
            if not gpry_match:
                return None
            
            gyro_pitch = float(gpry_match.group(1))
            gyro_roll = float(gpry_match.group(2))
            gyro_yaw = float(gpry_match.group(3))
            
            return DroneMetadata(
                timestamp=timestamp,
                datetime_str=datetime_str,
                home_longitude=home_lon,
                home_latitude=home_lat,
                gps_longitude=gps_lon,
                gps_latitude=gps_lat,
                gps_altitude=gps_alt,
                iso=iso,
                shutter_speed=shutter,
                exposure_value=ev,
                f_number=f_num,
                frame_pitch=frame_pitch,
                frame_roll=frame_roll,
                frame_yaw=frame_yaw,
                gyro_pitch=gyro_pitch,
                gyro_roll=gyro_roll,
                gyro_yaw=gyro_yaw
            )
        
        except (ValueError, AttributeError) as e:
            # Skip entries that can't be parsed
            return None
    
    def parse_srt_timestamp(self, time_str):
        """
        Convert SRT timestamp to seconds.
        
        Args:
            time_str: Timestamp in format "HH:MM:SS,mmm"
            
        Returns:
            Timestamp in seconds
        """
        time_str = time_str.replace(',', '.')
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def extract_all_metadata(self):
        """
        Extract all metadata entries from the video.
        
        Returns:
            List of DroneMetadata objects, one per second of video
        """
        srt_content = self.extract_subtitle_text()
        metadata_list = []
        
        # Split into individual subtitle entries
        entries = re.split(r'\n\n+', srt_content.strip())
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = entry.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # Parse timestamp from second line (e.g., "00:00:00,000 --> 00:00:01,000")
            timestamp_match = re.match(r'([\d:,]+)\s*-->\s*([\d:,]+)', lines[1])
            if not timestamp_match:
                continue
            
            start_time = self.parse_srt_timestamp(timestamp_match.group(1))
            
            # Parse the metadata text (everything after the timestamp line)
            metadata_text = '\n'.join(lines[2:])
            
            metadata = self.parse_metadata_entry(metadata_text, start_time)
            if metadata:
                metadata_list.append(metadata)
        
        return metadata_list
    
    def get_metadata_at_time(self, time_seconds):
        """
        Get the metadata closest to a specific time in the video.
        
        Args:
            time_seconds: Time in seconds from the start of the video
            
        Returns:
            DroneMetadata object or None if not found
        """
        all_metadata = self.extract_all_metadata()
        
        if not all_metadata:
            return None
        
        # Find the closest metadata entry
        closest = min(all_metadata, key=lambda m: abs(m.timestamp - time_seconds))
        return closest
    
    def export_to_csv(self, output_path: str):
        """
        Export all metadata to a CSV file.
        
        Args:
            output_path: Path to the output CSV file
        """
        import csv
        
        metadata_list = self.extract_all_metadata()
        
        if not metadata_list:
            print("No metadata found to export")
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'timestamp_sec', 'datetime', 
                'home_longitude', 'home_latitude',
                'gps_longitude', 'gps_latitude', 'gps_altitude_m',
                'iso', 'shutter_speed', 'exposure_value', 'f_number',
                'frame_pitch', 'frame_roll', 'frame_yaw',
                'gyro_pitch', 'gyro_roll', 'gyro_yaw'
            ])
            
            # Write data
            for metadata in metadata_list:
                writer.writerow([
                    metadata.timestamp, metadata.datetime_str,
                    metadata.home_longitude, metadata.home_latitude,
                    metadata.gps_longitude, metadata.gps_latitude, metadata.gps_altitude,
                    metadata.iso, metadata.shutter_speed, metadata.exposure_value, metadata.f_number,
                    metadata.frame_pitch, metadata.frame_roll, metadata.frame_yaw,
                    metadata.gyro_pitch, metadata.gyro_roll, metadata.gyro_yaw
                ])
        
        print(f"Exported {len(metadata_list)} metadata entries to {output_path}")