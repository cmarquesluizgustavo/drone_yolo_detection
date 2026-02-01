#!/usr/bin/env python3
"""
Video Preprocessing Pipeline for Person Detection

This script processes raw video files to create:
1. Video clips of X seconds (inputs)
2. Frame samples every W frames from clips (input_samples)

Parameters:
- X: Clip duration in seconds
- Z: Target resolution (e.g., '1080p', '720p', '480p')
- W: Frame sampling interval (extract 1 frame every W frames)
"""

import cv2
import argparse
from pathlib import Path
from typing import Tuple, Dict

# Resolution mapping
RESOLUTIONS = {
    '1080p': (1920, 1080),
    '720p': (1280, 720),
    '480p': (854, 480),
    '360p': (640, 360),
    '240p': (426, 240)
}

class VideoPreprocessor:
    def extract_samples_virtual_clips(self, video_path: str, clip_duration: int, frame_interval: int, rel_subdir: Path = Path("")) -> None:
        """Extract frames directly from video, organize into per-clip folders (virtual clips), skip errors and continue."""
        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Output directory for this video
        output_samples_subdir = self.samples_dir / rel_subdir
        output_samples_subdir.mkdir(parents=True, exist_ok=True)

        print(f"  Extracting frames from {video_name} (clip duration: {clip_duration}s, frame interval: {frame_interval} frames)")

        num_clips = int(duration // clip_duration)
        total_samples = 0
        for clip_idx in range(num_clips):
            clip_start_sec = clip_idx * clip_duration
            clip_end_sec = min((clip_idx + 1) * clip_duration, duration)
            clip_folder = output_samples_subdir / f"{video_name}_clip_{clip_idx:03d}_{clip_duration}s"
            clip_folder.mkdir(exist_ok=True)
            sample_idx = 0
            start_frame = int(clip_start_sec * fps)
            end_frame = int(clip_end_sec * fps)
            for frame_number in range(start_frame, end_frame, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"    [Warning] Could not read frame {frame_number} (t={frame_number/fps:.2f}s), skipping.")
                        continue
                    output_path = clip_folder / f"frame_{sample_idx:04d}.jpg"
                    try:
                        cv2.imwrite(str(output_path), frame)
                        sample_idx += 1
                        total_samples += 1
                    except Exception as e:
                        print(f"    [Warning] Error saving frame {frame_number} (t={frame_number/fps:.2f}s): {e}. Skipping.")
                except Exception as e:
                    print(f"    [Warning] Error extracting frame {frame_number} (t={frame_number/fps:.2f}s): {e}. Skipping.")
            print(f"    Extracted {sample_idx} frames to {clip_folder}")
        cap.release()
        print(f"    Total frames extracted for {video_name}: {total_samples}")
        
    def __init__(self, raw_dir: str, clips_dir: str, samples_dir: str):
        self.raw_dir = Path(raw_dir)
        self.clips_dir = Path(clips_dir)
        self.samples_dir = Path(samples_dir)
        
        # Create directories if they don't exist
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Compression presets
        self.compression_presets = {
            'no_compression': {},
            'high_quality': {'crf': 18, 'preset': 'slow'},
            'balanced': {'crf': 23, 'preset': 'medium'},
            'compressed': {'crf': 28, 'preset': 'fast'},
            'very_compressed': {'crf': 35, 'preset': 'faster'}
        }
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get basic video information."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
    
    def calculate_target_size(self, original_size: Tuple[int, int], target_resolution: str) -> Tuple[int, int]:
        """Calculate target size maintaining aspect ratio. If 'no_compression', return original size."""
        if target_resolution == 'no_compression':
            return original_size
        if target_resolution not in RESOLUTIONS:
            raise ValueError(f"Unsupported resolution: {target_resolution}. Choose from {list(RESOLUTIONS.keys())}")
        target_width, target_height = RESOLUTIONS[target_resolution]
        original_width, original_height = original_size
        # Calculate scaling factor to fit within target resolution
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        # Ensure even dimensions for video encoding
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        return new_width, new_height
    
    def extract_video_clips(self, video_path: str, clip_duration: int, target_resolution: str, 
                           compression_preset: str = 'balanced', target_fps: float = None, 
                           max_bitrate: str = None, rel_subdir: Path = Path("")) -> None:
        """Extract video clips of specified duration with compression options, preserving subfolder structure. If 'no_compression', just copy segments without re-encoding or resizing."""
        video_info = self.get_video_info(video_path)
        video_name = Path(video_path).stem

        print(f"Processing {video_name}:")
        print(f"  Original: {video_info['width']}x{video_info['height']}, {video_info['duration']:.1f}s, {video_info['fps']:.1f} fps")

        # Calculate target size
        target_size = self.calculate_target_size((video_info['width'], video_info['height']), target_resolution)

        # Handle FPS reduction
        original_fps = video_info['fps']
        if target_fps and target_fps < original_fps:
            output_fps = target_fps
            fps_reduction = original_fps / target_fps
            print(f"  Reducing FPS: {original_fps:.1f} → {output_fps:.1f} (saves ~{((fps_reduction-1)/fps_reduction*100):.0f}% size)")
        else:
            output_fps = original_fps
            fps_reduction = 1.0

        if compression_preset == 'no_compression':
            print(f"  Target: original resolution, no compression")
        else:
            print(f"  Target: {target_size[0]}x{target_size[1]}, {output_fps:.1f} fps")
            print(f"  Compression: {compression_preset}")
            if max_bitrate:
                print(f"  Max bitrate: {max_bitrate}")

        # Calculate number of clips
        num_clips = int(video_info['duration'] // clip_duration)
        if num_clips == 0:
            print(f"  Warning: Video too short for {clip_duration}s clips")
            return

        print(f"  Extracting {num_clips} clips of {clip_duration}s each")

        # Use ffmpeg for better compression or just copy
        import subprocess

        # Ensure output subdirectory exists
        output_clips_subdir = self.clips_dir / rel_subdir
        output_clips_subdir.mkdir(parents=True, exist_ok=True)

        for clip_idx in range(num_clips):
            start_time = clip_idx * clip_duration

            # Create descriptive filename with compression and fps info
            if compression_preset == 'no_compression':
                filename = f"{video_name}_clip_{clip_idx:03d}_original.mp4"
            else:
                fps_suffix = f"_{int(output_fps)}fps" if target_fps and target_fps < original_fps else ""
                bitrate_suffix = f"_{max_bitrate}" if max_bitrate else ""
                filename = f"{video_name}_clip_{clip_idx:03d}_{target_resolution}_{compression_preset}{fps_suffix}{bitrate_suffix}.mp4"
            output_path = output_clips_subdir / filename

            if compression_preset == 'no_compression':
                # ffmpeg copy mode: no re-encoding, no scaling
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', str(video_path),
                    '-t', str(clip_duration),
                    '-c', 'copy',
                    str(output_path)
                ]
            else:
                # Build ffmpeg command for better compression
                cmd = [
                    'ffmpeg', '-y',  # -y to overwrite output files
                    '-ss', str(start_time),  # Start time
                    '-i', str(video_path),   # Input file
                    '-t', str(clip_duration),  # Duration
                    '-vf', f'scale={target_size[0]}:{target_size[1]}',  # Scale
                    '-c:v', 'libx264',  # Use H.264 codec
                ]
                # Add compression settings
                preset_settings = self.compression_presets[compression_preset]
                cmd.extend(['-crf', str(preset_settings['crf'])])
                cmd.extend(['-preset', preset_settings['preset']])
                # Add FPS control
                if target_fps and target_fps < original_fps:
                    cmd.extend(['-r', str(target_fps)])
                # Add bitrate limit if specified
                if max_bitrate:
                    cmd.extend(['-maxrate', max_bitrate, '-bufsize', f"{int(max_bitrate.rstrip('kM')) * 2}k"])
                # Audio settings (compress audio too)
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
                # Output file
                cmd.append(str(output_path))

            try:
                # Run ffmpeg with suppressed output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Get file size
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"    Created: {output_path.name} ({file_size_mb:.1f}MB)")
            except subprocess.CalledProcessError as e:
                print(f"    Error creating {output_path.name}: {e}")
                print(f"    Command: {' '.join(cmd)}")
                # Fallback to OpenCV if ffmpeg fails
                self._extract_clip_opencv_fallback(video_path, clip_idx, clip_duration, target_size, output_fps, output_path)
    
    def _extract_clip_opencv_fallback(self, video_path: str, clip_idx: int, clip_duration: int, 
                                    target_size: tuple, output_fps: float, output_path: Path):
        """Fallback method using OpenCV if ffmpeg fails."""
        print(f"    Using OpenCV fallback for {output_path.name}")
        
        cap = cv2.VideoCapture(video_path)
        video_info = self.get_video_info(video_path)
        
        frames_per_clip = int(video_info['fps'] * clip_duration)
        start_frame = clip_idx * frames_per_clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, target_size)
        
        # Extract frames for this clip
        for frame_idx in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            resized_frame = cv2.resize(frame, target_size)
            out.write(resized_frame)
        
        out.release()
        cap.release()
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"    Created: {output_path.name} ({file_size_mb:.1f}MB) [OpenCV]")
    
    def extract_frame_samples(self, clip_path: str, frame_interval: int, rel_subdir: Path = Path("")) -> None:
        """Extract frame samples from video clips, preserving subfolder structure."""
        clip_name = Path(clip_path).stem

        cap = cv2.VideoCapture(clip_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output directory for this clip (includes compression info in name)
        output_samples_subdir = self.samples_dir / rel_subdir
        output_samples_subdir.mkdir(parents=True, exist_ok=True)
        output_dir = output_samples_subdir / f"{clip_name}_every{frame_interval}frames"
        output_dir.mkdir(exist_ok=True)

        frame_idx = 0
        sample_idx = 0

        print(f"  Extracting frames from {clip_name} (every {frame_interval} frames)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame every W frames
            if frame_idx % frame_interval == 0:
                output_path = output_dir / f"frame_{sample_idx:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                sample_idx += 1

            frame_idx += 1

        cap.release()
        print(f"    Extracted {sample_idx} frames to {output_dir}")
    
    def process_all_videos(self, clip_duration: int, frame_interval: int):
        """Process all videos, extract frames into per-clip folders (virtual clips), skipping errors."""
        print("=" * 70)
        print("VIDEO FRAME EXTRACTION PIPELINE (VIRTUAL CLIPS)")
        print("=" * 70)
        print(f"Parameters:")
        print(f"  Clip duration: {clip_duration} seconds")
        print(f"  Frame interval: {frame_interval} frames")
        print("=" * 70)

        # Recursively get all video files
        video_files = list(self.raw_dir.rglob("*.mp4"))
        if not video_files:
            print("No MP4 files found in raw directory!")
            return

        print(f"\nFound {len(video_files)} video files")

        print("\nSTEP: Extracting frame samples from videos into virtual clips...")
        for video_file in video_files:
            rel_subdir = video_file.parent.relative_to(self.raw_dir)
            self.extract_samples_virtual_clips(str(video_file), clip_duration, frame_interval, rel_subdir)

        print("\n" + "=" * 70)
        print("FRAME EXTRACTION COMPLETE!")
        print(f"  Frame samples: {len(list(self.samples_dir.rglob('*_clip_*')))} directories")
        print(f"  Location: {self.samples_dir}")
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Video preprocessing pipeline for person detection")
    parser.add_argument("-X", "--clip-duration", type=int, default=10,
                       help="Clip duration in seconds (default: 10)")
    parser.add_argument("-Z", "--resolution", type=str, default="720p", 
                       choices=list(RESOLUTIONS.keys()),
                       help="Target resolution (default: 720p)")
    parser.add_argument("-W", "--frame-interval", type=int, default=30,
                       help="Frame sampling interval - extract 1 frame every W frames (default: 30)")
    
    # Compression parameters
    parser.add_argument("-C", "--compression", type=str, default="balanced",
                       choices=['high_quality', 'balanced', 'compressed', 'very_compressed'],
                       help="Compression preset: high_quality, balanced, compressed, very_compressed (default: balanced)")
    parser.add_argument("--no-compression", action="store_true", help="If set, disables compression and resizing, keeping original video quality and resolution.")
    parser.add_argument("-F", "--fps", type=float, default=None,
                       help="Target FPS (reduces from original if lower, saves space)")
    parser.add_argument("-B", "--max-bitrate", type=str, default=None,
                       help="Maximum bitrate (e.g., '2M', '1000k') - limits file size")
    
    # Directory parameters
    parser.add_argument("--raw", type=str, default="inputs/raw",
                       help="Raw videos directory (default: inputs/raw)")
    parser.add_argument("--clips", type=str, default="inputs/clips",
                       help="Processed clips directory (default: inputs/clips)")
    parser.add_argument("--samples", type=str, default="inputs/samples",
                       help="Frame samples directory (default: inputs/samples)")
    
    args = parser.parse_args()
    
    # If no-compression flag is set, override related args
    if args.no_compression:
        args.compression = 'no_compression'
        args.resolution = 'no_compression'
        args.fps = None
        args.max_bitrate = None

    # Validate bitrate format if provided
    if args.max_bitrate and not (args.max_bitrate.endswith('k') or args.max_bitrate.endswith('M')):
        print("Error: Bitrate must end with 'k' or 'M' (e.g., '1000k', '2M')")
        return

    # Create preprocessor (clips_dir is not used, but kept for compatibility)
    preprocessor = VideoPreprocessor(args.raw, "unused_clips_dir", args.samples)

    # Process all videos (extract frames into virtual clips)
    preprocessor.process_all_videos(args.clip_duration, args.frame_interval)

if __name__ == "__main__":
    main()