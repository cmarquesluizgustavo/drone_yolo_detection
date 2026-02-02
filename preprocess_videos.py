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
            'none': {'crf': None, 'preset': None},  # No compression - copy codec
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
                           max_bitrate: str = None, relative_path: str = None) -> None:
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
            
            # Maintain directory structure
            if relative_path:
                output_dir = self.clips_dir / relative_path
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / filename
            else:
                output_path = self.clips_dir / filename
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output files
                '-ss', str(start_time),  # Start time
                '-i', str(video_path),   # Input file
                '-t', str(clip_duration),  # Duration
            ]
            
            # Add compression settings
            preset_settings = self.compression_presets[compression_preset]
            
            # Check if we need to re-encode (can't use copy with filters or fps change)
            needs_reencode = (
                preset_settings['crf'] is not None or  # Compression requested
                target_fps and target_fps < original_fps or  # FPS change requested
                target_size != (video_info['width'], video_info['height'])  # Resolution change
            )
            
            if needs_reencode:
                # Need to re-encode
                cmd.extend(['-vf', f'scale={target_size[0]}:{target_size[1]}'])  # Scale
                
                if preset_settings['crf'] is not None:
                    # Apply compression
                    cmd.extend(['-c:v', 'libx264'])
                    cmd.extend(['-crf', str(preset_settings['crf'])])
                    cmd.extend(['-preset', preset_settings['preset']])
                else:
                    # No compression but still need to encode due to filters/fps
                    cmd.extend(['-c:v', 'libx264'])
                    cmd.extend(['-crf', '18'])  # High quality
                    cmd.extend(['-preset', 'medium'])
                
                # Add FPS control
                if target_fps and target_fps < original_fps:
                    cmd.extend(['-r', str(target_fps)])
                
                # Add bitrate limit if specified
                if max_bitrate:
                    cmd.extend(['-maxrate', max_bitrate, '-bufsize', f"{int(max_bitrate.rstrip('kM')) * 2}k"])
                
                # Compress audio
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            else:
                # Can use codec copy (no filters, no fps change, no resolution change)
                cmd.extend(['-c:v', 'copy'])
                cmd.extend(['-c:a', 'copy'])
            
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
    
    def extract_frame_samples(self, clip_path: str, frame_interval: int, relative_path: str = None) -> None:
        """Extract frame samples from video clips."""
        clip_name = Path(clip_path).stem
        
        cap = cv2.VideoCapture(clip_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory for this clip (includes compression info in name)
        # Maintain directory structure
        sample_folder_name = f"{clip_name}_every{frame_interval}frames"
        if relative_path:
            output_dir = self.samples_dir / relative_path / sample_folder_name
        else:
            output_dir = self.samples_dir / sample_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def process_clips_to_frames(self, frame_interval: int):
        """Process existing clips directly to frame samples without raw videos."""
        print("=" * 70)
        print("CLIPS TO FRAMES PIPELINE")
        print("=" * 70)
        print(f"Parameters:")
        print(f"  Frame interval (W): every {frame_interval} frames")
        print(f"  Source: {self.clips_dir}")
        print(f"  Output: {self.samples_dir}")
        print("=" * 70)
        
        # Get all clip files
        clip_files = list(self.clips_dir.rglob("*.mp4")) + list(self.clips_dir.rglob("*.MP4"))
        if not clip_files:
            print(f"\nNo MP4 files found in {self.clips_dir}!")
            return
        
        print(f"\nFound {len(clip_files)} clip files")
        
        # Extract frame samples
        print("\nExtracting frame samples from clips...")
        for clip_file in clip_files:
            # Calculate relative path to maintain structure
            relative_path = clip_file.relative_to(self.clips_dir).parent
            self.extract_frame_samples(str(clip_file), frame_interval, str(relative_path))
        
        # Count sample directories
        sample_dirs = [d for d in self.samples_dir.rglob("*") if d.is_dir()]
        
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print(f"Results:")
        print(f"  Processed clips: {len(clip_files)} files")
        print(f"  Frame sample directories: {len(sample_dirs)}")
        print(f"  Location: {self.samples_dir}")
        print("=" * 70)
    
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
        
        # Get all video files recursively
        video_files = list(self.raw_dir.rglob("*.MP4")) + list(self.raw_dir.rglob("*.mp4"))
        if not video_files:
            print("No MP4 files found in raw directory!")
            return
        
        print(f"\nFound {len(video_files)} video files")
        
        # Step 1: Extract clips
        print("\nSTEP 1: Extracting video clips...")
        total_size_mb = 0
        for video_file in video_files:
            # Calculate relative path to maintain structure
            relative_path = video_file.relative_to(self.raw_dir).parent
            self.extract_video_clips(str(video_file), clip_duration, target_resolution, 
                                   compression_preset, target_fps, max_bitrate, str(relative_path))
        
        # Calculate total size of clips
        clip_files = list(self.clips_dir.rglob("*.mp4"))
        for clip_file in clip_files:
            total_size_mb += clip_file.stat().st_size / (1024 * 1024)
        
        # Step 2: Extract frame samples
        print("\nSTEP 2: Extracting frame samples...")
        for clip_file in clip_files:
            # Calculate relative path to maintain structure
            relative_path = clip_file.relative_to(self.clips_dir).parent
            self.extract_frame_samples(str(clip_file), frame_interval, str(relative_path))
        
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print(f"Results:")
        print(f"  Video clips: {len(clip_files)} files ({total_size_mb:.1f}MB total)")
        print(f"  Frame samples: {len(list(self.samples_dir.iterdir()))} directories")
        print(f"  Location: {self.clips_dir} and {self.samples_dir}")
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Video preprocessing pipeline for person detection")
    
    # Mode selection
    parser.add_argument("--clips-only", action="store_true",
                       help="Process existing clips to frames only, skip raw video processing")
    
    parser.add_argument("-X", "--clip-duration", type=int, default=10,
                       help="Clip duration in seconds (default: 10)")
    parser.add_argument("-Z", "--resolution", type=str, default="1080p", 
                       choices=list(RESOLUTIONS.keys()),
                       help="Target resolution (default: 1080p)")
    parser.add_argument("-W", "--frame-interval", type=int, default=10,
                       help="Frame sampling interval - extract 1 frame every W frames (default: 10)")

    # Compression parameters
    parser.add_argument("-C", "--compression", type=str, default="none",
                       choices=['none', 'high_quality', 'balanced', 'compressed', 'very_compressed'],
                       help="Compression preset: none (no compression), high_quality, balanced, compressed, very_compressed (default: balanced)")
    parser.add_argument("-F", "--fps", type=int, default=10,
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
    
    # Choose processing mode
    if args.clips_only:
        # Process existing clips to frames only
        preprocessor.process_clips_to_frames(args.frame_interval)
    else:
        # Process all videos (clips + frames)
        preprocessor.process_all_videos(args.clip_duration, args.resolution, args.frame_interval,
                                      args.compression, args.fps, args.max_bitrate)

if __name__ == "__main__":
    main()