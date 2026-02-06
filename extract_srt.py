import re
import os
import subprocess
from pathlib import Path


def extract_srt_from_videos(inputs_dir: str = 'inputs'):
    """Extract SRT subtitle files from all MP4 videos in the inputs directory."""
    processed_count = 0
    failed_count = 0
    
    # Walk through all subdirectories and files in the inputs directory
    for root, dirs, files in os.walk(inputs_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_path = os.path.join(root, file)
                srt_path = os.path.splitext(mp4_path)[0] + '.srt'
                
                # Skip if SRT already exists
                if os.path.exists(srt_path):
                    print(f'SRT already exists, skipping: {srt_path}')
                    continue
                
                # ffmpeg command to extract subtitles
                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output file if exists
                    '-i', mp4_path,
                    '-map', '0:s:0',  # First subtitle stream
                    srt_path
                ]
                print(f'Extracting subtitles from: {mp4_path}')
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f'  → Subtitles saved to: {srt_path}')
                    processed_count += 1
                except subprocess.CalledProcessError as e:
                    print(f'  ✗ No subtitles found or extraction failed for: {mp4_path}')
                    failed_count += 1
                    # Optionally print error: print(e.stderr.decode())
    
    print(f"\n✓ Extracted {processed_count} SRT files, {failed_count} failed")
    return processed_count, failed_count


def extract_first_values_from_srt(srt_path: str) -> dict:
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {}

    def _apply_dir(value: str, direction: str) -> float:
        direction = str(direction).upper().strip()
        v = float(value)
        if direction in ('W', 'S'):
            return -abs(v)
        return abs(v)

    # Prefer HOME() coordinates if present; otherwise fall back to GPS().
    # Note: DJI-style subtitles use N/S/E/W as *direction*, not sign.
    home_pattern = r'HOME\((?P<lon_dir>[EW]):\s*(?P<lon>[\d.]+),\s*(?P<lat_dir>[NS]):\s*(?P<lat>[\d.]+)\)'
    home_match = re.search(home_pattern, content)
    if home_match:
        result['lon'] = _apply_dir(home_match.group('lon'), home_match.group('lon_dir'))
        result['lat'] = _apply_dir(home_match.group('lat'), home_match.group('lat_dir'))
    else:
        gps_pattern = r'GPS\((?P<lon_dir>[EW]):\s*(?P<lon>[\d.]+),\s*(?P<lat_dir>[NS]):\s*(?P<lat>[\d.]+),\s*[-\d.]+m?\)'
        gps_match = re.search(gps_pattern, content)
        if gps_match:
            result['lon'] = _apply_dir(gps_match.group('lon'), gps_match.group('lon_dir'))
            result['lat'] = _apply_dir(gps_match.group('lat'), gps_match.group('lat_dir'))
    
    # Extract first G.PRY (gimbal pitch, roll, yaw)
    pry_pattern = r'G\.PRY\s*\(\s*([-\d.]+)°,\s*([-\d.]+)°,\s*([-\d.]+)°\)'
    pry_match = re.search(pry_pattern, content)
    
    if pry_match:
        result['pitch'] = float(pry_match.group(1))
        result['roll'] = float(pry_match.group(2))
        result['yaw'] = float(pry_match.group(3))
    
    return result


def save_values_to_txt(values: dict, txt_path: str):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"pitch: {values.get('pitch', 'N/A')}\n")
        f.write(f"roll: {values.get('roll', 'N/A')}\n")
        f.write(f"yaw: {values.get('yaw', 'N/A')}\n")
        f.write(f"lat: {values.get('lat', 'N/A')}\n")
        f.write(f"lon: {values.get('lon', 'N/A')}\n")


def process_all_srt_files(raw_dir: str = 'inputs/raw'):
    raw_path = Path(raw_dir)
    processed_count = 0
    
    # Iterate through all SRT files
    for srt_file in raw_path.rglob('*.srt'):
        print(f"Processing: {srt_file}")
        
        # Extract first values
        values = extract_first_values_from_srt(str(srt_file))
        
        # Create TXT file path (same location and name as SRT, but .txt extension)
        txt_file = srt_file.with_suffix('.txt')
        
        # Save to TXT file
        save_values_to_txt(values, str(txt_file))
        
        print(f"  → Saved to: {txt_file}")
        print(f"     pitch={values.get('pitch', 'N/A')}, roll={values.get('roll', 'N/A')}, "
              f"yaw={values.get('yaw', 'N/A')}, lat={values.get('lat', 'N/A')}, lon={values.get('lon', 'N/A')}")
        
        processed_count += 1
    
    print(f"\n✓ Processed {processed_count} SRT files")


def main():
    """Main function to extract SRT files from videos and then extract first values."""
    print("=" * 70)
    print("SRT Extraction and Processing Pipeline")
    print("=" * 70)
    print()
    
    # Step 1: Extract SRT files from MP4 videos
    print("STEP 1: Extracting SRT files from videos...")
    print("-" * 70)
    extract_srt_from_videos()
    print()
    
    # Step 2: Extract first values from SRT files
    print("STEP 2: Extracting first gimbal pitch, roll, yaw, lat, lon from each SRT file...")
    print("-" * 70)
    process_all_srt_files()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
