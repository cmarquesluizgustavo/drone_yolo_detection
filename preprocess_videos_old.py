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
    def __init__(self, raw_dir: str, clips_dir: str, samples_dir: str):
        self.raw_dir = Path(raw_dir)
        self.clips_dir = Path(clips_dir)
        self.samples_dir = Path(samples_dir)
        
        # Create directories if they don't exist
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Compression presets
        self.compression_presets = {
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
        """Calculate target size maintaining aspect ratio."""
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
                           max_bitrate: str = None) -> None:
        """Extract video clips of specified duration with compression options."""
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
        
        # Use ffmpeg for better compression
        import subprocess
        
        for clip_idx in range(num_clips):
            start_time = clip_idx * clip_duration
            
            # Create descriptive filename with compression and fps info
            fps_suffix = f"_{int(output_fps)}fps" if target_fps and target_fps < original_fps else ""
            bitrate_suffix = f"_{max_bitrate}" if max_bitrate else ""
            filename = f"{video_name}_clip_{clip_idx:03d}_{target_resolution}_{compression_preset}{fps_suffix}{bitrate_suffix}.mp4"
            output_path = self.clips_dir / filename
            
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
    
    def extract_frame_samples(self, clip_path: str, frame_interval: int) -> None:
        """Extract frame samples from video clips."""
        clip_name = Path(clip_path).stem
        
        cap = cv2.VideoCapture(clip_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory for this clip (includes compression info in name)
        output_dir = self.samples_dir / f"{clip_name}_every{frame_interval}frames"
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
        print(f"    Extracted {sample_idx} frames to {output_dir.name}/")
    
    def process_all_videos(self, clip_duration: int, target_resolution: str, frame_interval: int,
                          compression_preset: str = 'balanced', target_fps: float = None, 
                          max_bitrate: str = None):
        """Process all videos in the raw_inputs directory."""
        print("=" * 70)
        print("VIDEO PREPROCESSING PIPELINE")
        print("=" * 70)
        print(f"Parameters:")
        print(f"  Clip duration (X): {clip_duration} seconds")
        print(f"  Target resolution (Z): {target_resolution}")
        print(f"  Frame interval (W): every {frame_interval} frames")
        print(f"  Compression preset: {compression_preset}")
        if target_fps:
            print(f"  Target FPS: {target_fps}")
        if max_bitrate:
            print(f"  Max bitrate: {max_bitrate}")
        print("=" * 70)
        
        # Get all video files
        video_files = list(self.raw_dir.glob("*.mp4"))
        if not video_files:
            print("No MP4 files found in raw directory!")
            return
        
        print(f"\nFound {len(video_files)} video files")
        
        # Step 1: Extract clips
        print("\nSTEP 1: Extracting video clips...")
        total_size_mb = 0
        for video_file in video_files:
            self.extract_video_clips(str(video_file), clip_duration, target_resolution, 
                                   compression_preset, target_fps, max_bitrate)
        
        # Calculate total size of clips
        clip_files = list(self.clips_dir.glob("*.mp4"))
        for clip_file in clip_files:
            total_size_mb += clip_file.stat().st_size / (1024 * 1024)
        
        # Step 2: Extract frame samples
        print("\nSTEP 2: Extracting frame samples...")
        for clip_file in clip_files:
            self.extract_frame_samples(str(clip_file), frame_interval)
        
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print(f"Results:")
        print(f"  Video clips: {len(clip_files)} files ({total_size_mb:.1f}MB total)")
        print(f"  Frame samples: {len(list(self.samples_dir.iterdir()))} directories")
        print(f"  Location: {self.clips_dir} and {self.samples_dir}")
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
    
    # Validate bitrate format if provided
    if args.max_bitrate and not (args.max_bitrate.endswith('k') or args.max_bitrate.endswith('M')):
        print("Error: Bitrate must end with 'k' or 'M' (e.g., '1000k', '2M')")
        return
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(args.raw, args.clips, args.samples)
    
    # Process all videos
    preprocessor.process_all_videos(args.clip_duration, args.resolution, args.frame_interval,
                                  args.compression, args.fps, args.max_bitrate)

if __name__ == "__main__":
    main()