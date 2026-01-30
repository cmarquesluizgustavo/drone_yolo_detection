#!/usr/bin/env python3
"""
Dual-Drone Video Preprocessing Pipeline

This script processes videos from two drones (drone1 and drone2) to extract frame samples.
Since videos are already 1080p, no compression or resizing is performed.

Parameters:
- X: Clip duration in seconds (for organizing frames)
- W: Frame sampling interval (extract 1 frame every W frames)

Input structure:
  inputs/
    drone1/
      video1.MP4
      video2.MP4
    drone2/
      video1.MP4
      video2.MP4

Output structure:
  inputs/
    drone1_frames/
      video1_clip_000/
        frame_0000.jpg
        frame_0001.jpg
      video1_clip_001/
    drone2_frames/
"""

import cv2
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import os
import sys
import warnings
import json
from drone_metadata_extractor import DroneMetadataExtractor, DroneMetadata

# Suppress all OpenCV/FFmpeg warnings about corrupted frames
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Redirect stderr to devnull to suppress FFmpeg errors
class SuppressStderr:
    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.old_stderr

class DualDronePreprocessor:
    def __init__(self, input_dir: str, output_base_dir: str = None, extract_metadata: bool = True):
        self.input_dir = Path(input_dir)
        self.extract_metadata = extract_metadata
        
        # Set up input directories
        self.drone1_dir = self.input_dir / "drone1"
        self.drone2_dir = self.input_dir / "drone2"
        
        # Set up output directories
        if output_base_dir is None:
            output_base_dir = self.input_dir
        else:
            output_base_dir = Path(output_base_dir)
        
        self.drone1_frames_dir = output_base_dir / "drone1_frames"
        self.drone2_frames_dir = output_base_dir / "drone2_frames"
        
        # Verify input directories exist
        if not self.drone1_dir.exists():
            raise FileNotFoundError(f"Drone1 directory not found: {self.drone1_dir}")
        if not self.drone2_dir.exists():
            raise FileNotFoundError(f"Drone2 directory not found: {self.drone2_dir}")
        
        # Create output directories
        self.drone1_frames_dir.mkdir(parents=True, exist_ok=True)
        self.drone2_frames_dir.mkdir(parents=True, exist_ok=True)
    
    def save_metadata_to_json(self, metadata: DroneMetadata, output_path: Path):
        """Save metadata to a JSON file alongside the frame."""
        metadata_dict = {
            'timestamp_sec': metadata.timestamp,
            'datetime': metadata.datetime_str,
            'home': {
                'longitude': metadata.home_longitude,
                'latitude': metadata.home_latitude
            },
            'gps': {
                'longitude': metadata.gps_longitude,
                'latitude': metadata.gps_latitude,
                'altitude_m': metadata.gps_altitude
            },
            'camera': {
                'iso': metadata.iso,
                'shutter_speed': metadata.shutter_speed,
                'exposure_value': metadata.exposure_value,
                'f_number': metadata.f_number
            },
            'orientation': {
                'frame_pitch': metadata.frame_pitch,
                'frame_roll': metadata.frame_roll,
                'frame_yaw': metadata.frame_yaw,
                'gyro_pitch': metadata.gyro_pitch,
                'gyro_roll': metadata.gyro_roll,
                'gyro_yaw': metadata.gyro_yaw
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get basic video information."""
        # Suppress all FFmpeg/OpenCV output
        cv2.setLogLevel(0)
        
        with SuppressStderr():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
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
    
    def extract_frames_from_video(self, video_path: str, output_dir: Path, 
                                  clip_duration: int, frame_interval: int,
                                  start_time: float = 0, end_time: float = None) -> int:
        """
        Extract frames from a video at specified intervals.
        
        Args:
            video_path: Path to the input video
            output_dir: Base directory for output frames
            clip_duration: Duration in seconds to group frames into clips
            frame_interval: Extract 1 frame every W frames
            start_time: Start time in seconds (default: 0)
            end_time: End time in seconds (default: None = until end)
            
        Returns:
            Total number of frames extracted
        """
        video_info = self.get_video_info(video_path)
        video_name = Path(video_path).stem
        
        print(f"\n  Processing {video_name}:")
        print(f"    Resolution: {video_info['width']}x{video_info['height']}")
        print(f"    Duration: {video_info['duration']:.1f}s")
        print(f"    FPS: {video_info['fps']:.1f}")
        print(f"    Total frames: {video_info['frame_count']}")
        
        # Initialize metadata extractor if enabled
        metadata_extractor = None
        all_metadata = []
        if self.extract_metadata:
            try:
                print(f"    Extracting metadata from video...")
                metadata_extractor = DroneMetadataExtractor(video_path)
                all_metadata = metadata_extractor.extract_all_metadata()
                print(f"    ✓ Found {len(all_metadata)} metadata entries")
            except Exception as e:
                print(f"    ⚠ Could not extract metadata: {e}")
                self.extract_metadata = False  # Disable for remaining videos
        
        # Calculate frame range
        start_frame = int(start_time * video_info['fps'])
        if end_time:
            end_frame = min(int(end_time * video_info['fps']), video_info['frame_count'])
        else:
            end_frame = video_info['frame_count']
        
        # Calculate number of clips
        duration_to_process = (end_frame - start_frame) / video_info['fps']
        num_clips = int(duration_to_process / clip_duration) + 1
        
        print(f"    Extracting frames every {frame_interval} frames")
        print(f"    Organizing into {num_clips} clips of ~{clip_duration}s each")
        
        # Suppress OpenCV/FFmpeg error messages for corrupted frames
        cv2.setLogLevel(0)
        
        # Open video with error suppression
        with SuppressStderr():
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        total_extracted = 0
        current_clip = 0
        frames_per_clip = int(video_info['fps'] * clip_duration)
        
        # Create initial clip directory
        clip_dir = output_dir / f"{video_name}_clip_{current_clip:03d}"
        clip_dir.mkdir(exist_ok=True)
        frames_in_clip = 0
        
        # For progress indication
        expected_frames = (end_frame - start_frame) // frame_interval
        corrupted_frames = 0
        
        while frame_idx < end_frame:
            # Read frame with error suppression
            with SuppressStderr():
                ret, frame = cap.read()
            
            # Handle corrupted/unreadable frames
            if not ret or frame is None:
                corrupted_frames += 1
                frame_idx += 1
                frames_in_clip += 1
                continue
            
            # Check if we need to start a new clip directory
            if frames_in_clip >= frames_per_clip:
                current_clip += 1
                clip_dir = output_dir / f"{video_name}_clip_{current_clip:03d}"
                clip_dir.mkdir(exist_ok=True)
                frames_in_clip = 0
            
            # Extract frame at interval
            if (frame_idx - start_frame) % frame_interval == 0:
                # Calculate the timestamp for this frame
                frame_time = frame_idx / video_info['fps']
                
                # If current frame is valid, use it
                if ret and frame is not None and frame.size > 0:
                    output_path = clip_dir / f"frame_{total_extracted:04d}.jpg"
                    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Save metadata if available
                    if all_metadata:
                        # Find closest metadata entry to this frame time
                        closest_metadata = min(all_metadata, key=lambda m: abs(m.timestamp - frame_time))
                        metadata_path = clip_dir / f"frame_{total_extracted:04d}.json"
                        self.save_metadata_to_json(closest_metadata, metadata_path)
                    
                    total_extracted += 1
                else:
                    # Try to get next valid frame (up to 10 attempts)
                    found_valid = False
                    for attempt in range(10):
                        with SuppressStderr():
                            ret_next, frame_next = cap.read()
                        
                        if ret_next and frame_next is not None and frame_next.size > 0:
                            output_path = clip_dir / f"frame_{total_extracted:04d}.jpg"
                            cv2.imwrite(str(output_path), frame_next, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            
                            # Save metadata if available (for the replacement frame)
                            if all_metadata:
                                replacement_time = (frame_idx + attempt) / video_info['fps']
                                closest_metadata = min(all_metadata, key=lambda m: abs(m.timestamp - replacement_time))
                                metadata_path = clip_dir / f"frame_{total_extracted:04d}.json"
                                self.save_metadata_to_json(closest_metadata, metadata_path)
                            
                            total_extracted += 1
                            found_valid = True
                            corrupted_frames += attempt
                            frame_idx += attempt
                            frames_in_clip += attempt
                            break
                        else:
                            corrupted_frames += 1
                    
                    if not found_valid:
                        # Skip this sampling point if no valid frame found
                        corrupted_frames += 1
            
            frame_idx += 1
            frames_in_clip += 1
        
        cap.release()
        
        # Report results
        success_msg = f"✓ Extracted {total_extracted} frames into {current_clip + 1} clip folders"
        if corrupted_frames > 0:
            success_msg += f" ({corrupted_frames} corrupted frames skipped)"
        print(f"    {success_msg}")
        
        return total_extracted
    
    def process_drone_videos(self, drone_dir: Path, output_dir: Path, 
                            clip_duration: int, frame_interval: int) -> Dict:
        """Process all videos for one drone."""
        # Get all video files (case-insensitive)
        video_files = list(drone_dir.glob("*.MP4")) + list(drone_dir.glob("*.mp4"))
        
        if not video_files:
            print(f"  ⚠ No video files found in {drone_dir}")
            return {'videos': 0, 'frames': 0}
        
        video_files.sort()  # Sort for consistent processing order
        
        print(f"\n  Found {len(video_files)} video files")
        
        total_frames = 0
        for video_file in video_files:
            frames_extracted = self.extract_frames_from_video(
                str(video_file), 
                output_dir, 
                clip_duration, 
                frame_interval
            )
            total_frames += frames_extracted
        
        return {
            'videos': len(video_files),
            'frames': total_frames,
            'clips': len(list(output_dir.glob("*_clip_*")))
        }
    
    def process_all(self, clip_duration: int = 10, frame_interval: int = 30,
                   skip_drone1: bool = False, skip_drone2: bool = False):
        """Process videos from both drones."""
        print("=" * 70)
        print("DUAL-DRONE VIDEO PREPROCESSING PIPELINE")
        print("=" * 70)
        print(f"Parameters:")
        print(f"  Clip duration (X): {clip_duration} seconds")
        print(f"  Frame interval (W): every {frame_interval} frames")
        print(f"  Metadata extraction: {'Enabled' if self.extract_metadata else 'Disabled'}")
        print(f"  Compression: None (maintaining original 1080p quality)")
        print(f"  JPEG quality: 95%")
        print("=" * 70)
        print(f"\nInput directories:")
        print(f"  Drone1: {self.drone1_dir}")
        print(f"  Drone2: {self.drone2_dir}")
        print(f"\nOutput directories:")
        print(f"  Drone1 frames: {self.drone1_frames_dir}")
        print(f"  Drone2 frames: {self.drone2_frames_dir}")
        print("=" * 70)
        
        results = {}
        
        # Process Drone 1
        if not skip_drone1:
            print("\n" + "=" * 70)
            print("PROCESSING DRONE 1")
            print("=" * 70)
            results['drone1'] = self.process_drone_videos(
                self.drone1_dir,
                self.drone1_frames_dir,
                clip_duration,
                frame_interval
            )
        else:
            print("\n⏭  Skipping Drone 1 (--skip-drone1)")
        
        # Process Drone 2
        if not skip_drone2:
            print("\n" + "=" * 70)
            print("PROCESSING DRONE 2")
            print("=" * 70)
            results['drone2'] = self.process_drone_videos(
                self.drone2_dir,
                self.drone2_frames_dir,
                clip_duration,
                frame_interval
            )
        else:
            print("\n⏭  Skipping Drone 2 (--skip-drone2)")
        
        # Print summary
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print("=" * 70)
        
        if 'drone1' in results:
            print(f"\nDrone 1:")
            print(f"  Videos processed: {results['drone1']['videos']}")
            print(f"  Frames extracted: {results['drone1']['frames']}")
            print(f"  Clip folders: {results['drone1']['clips']}")
            print(f"  Location: {self.drone1_frames_dir}")
        
        if 'drone2' in results:
            print(f"\nDrone 2:")
            print(f"  Videos processed: {results['drone2']['videos']}")
            print(f"  Frames extracted: {results['drone2']['frames']}")
            print(f"  Clip folders: {results['drone2']['clips']}")
            print(f"  Location: {self.drone2_frames_dir}")
        
        total_frames = sum(r['frames'] for r in results.values())
        total_videos = sum(r['videos'] for r in results.values())
        
        print(f"\nTotal:")
        print(f"  Videos: {total_videos}")
        print(f"  Frames: {total_frames}")
        print("=" * 70)
        
        return results
    
    def clean_output_directories(self):
        """Remove all extracted frames (useful for re-processing)."""
        print("\n⚠ Cleaning output directories...")
        
        for dir_to_clean in [self.drone1_frames_dir, self.drone2_frames_dir]:
            if dir_to_clean.exists():
                shutil.rmtree(dir_to_clean)
                dir_to_clean.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Cleaned: {dir_to_clean}")
        
        print("  ✓ Output directories cleaned\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dual-drone video preprocessing pipeline - extract frames from drone1 and drone2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 1 frame every 30 frames, organize in 10s clips
  python preprocess_videos.py -X 10 -W 30
  
  # Extract every frame (no skipping)
  python preprocess_videos.py -X 10 -W 1
  
  # Extract 1 frame per second (assuming 60fps video)
  python preprocess_videos.py -X 10 -W 60
  
  # Extract frames with metadata
  python preprocess_videos.py -X 10 -W 60 --extract-metadata
  
  # Extract frames without metadata (faster)
  python preprocess_videos.py -X 10 -W 60 --no-metadata
  
  # Process only drone1
  python preprocess_videos.py --skip-drone2
  
  # Clean and re-process
  python preprocess_videos.py --clean -X 10 -W 30
        """
    )
    
    parser.add_argument("-X", "--clip-duration", type=int, default=10,
                       help="Clip duration in seconds for organizing frames (default: 10)")
    parser.add_argument("-W", "--frame-interval", type=int, default=30,
                       help="Frame sampling interval - extract 1 frame every W frames (default: 30)")
    
    # Directory parameters
    parser.add_argument("--input-dir", type=str, default="inputs",
                       help="Input directory containing drone1/ and drone2/ folders (default: inputs)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output base directory (default: same as input-dir)")
    
    # Metadata extraction
    parser.add_argument("--extract-metadata", action="store_true", default=True,
                       help="Extract drone metadata (GPS, orientation, camera settings) - default: enabled")
    parser.add_argument("--no-metadata", action="store_true",
                       help="Skip metadata extraction (faster processing)")
    
    # Processing options
    parser.add_argument("--skip-drone1", action="store_true",
                       help="Skip processing drone1 videos")
    parser.add_argument("--skip-drone2", action="store_true",
                       help="Skip processing drone2 videos")
    parser.add_argument("--clean", action="store_true",
                       help="Clean output directories before processing")
    
    args = parser.parse_args()
    
    # Resolve metadata extraction flag
    extract_metadata = args.extract_metadata and not args.no_metadata
    
    try:
        # Create preprocessor
        preprocessor = DualDronePreprocessor(args.input_dir, args.output_dir, extract_metadata)
        
        # Clean if requested
        if args.clean:
            preprocessor.clean_output_directories()
        
        # Process all videos
        preprocessor.process_all(
            clip_duration=args.clip_duration,
            frame_interval=args.frame_interval,
            skip_drone1=args.skip_drone1,
            skip_drone2=args.skip_drone2
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure your directory structure is:")
        print("  inputs/")
        print("    drone1/")
        print("      video1.MP4")
        print("      video2.MP4")
        print("    drone2/")
        print("      video1.MP4")
        print("      video2.MP4")
        return 1
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
