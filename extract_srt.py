import os
import subprocess

# Set the root directory for input videos
INPUTS_DIR = os.path.join(os.path.dirname(__file__), 'inputs')

# Walk through all subdirectories and files in the inputs directory
for root, dirs, files in os.walk(INPUTS_DIR):
    for file in files:
        if file.lower().endswith('.mp4'):
            mp4_path = os.path.join(root, file)
            srt_path = os.path.splitext(mp4_path)[0] + '.srt'
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
                print(f'Subtitles saved to: {srt_path}')
            except subprocess.CalledProcessError as e:
                print(f'No subtitles found or extraction failed for: {mp4_path}')
                # Optionally print error: print(e.stderr.decode())
