#!/usr/bin/env python3
"""
Main script to run people detection pipeline.
"""

import argparse
import io
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection_pipeline import DetectionPipeline
from dual_drone_pipeline import DualDroneDetectionPipeline


def _parse_log_level(level: str) -> int:
    if not level:
        return logging.INFO
    level_name = str(level).upper().strip()
    return getattr(logging, level_name, logging.INFO)

class _TeeTextIO(io.TextIOBase):
    """Write text to two streams (e.g., console + file)."""

    def __init__(self, stream_a, stream_b):
        super().__init__()
        self._a = stream_a
        self._b = stream_b

    def write(self, s):
        n1 = self._a.write(s)
        self._b.write(s)
        return n1

    def flush(self):
        try:
            self._a.flush()
        finally:
            self._b.flush()


def _tee_console_to_file(console_path: Path):
    """Duplicate stdout/stderr to a file. Returns (restore_fn, file_handle)."""
    console_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(console_path, "a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeTextIO(old_out, fh)
    sys.stderr = _TeeTextIO(old_err, fh)

    def _restore():
        sys.stdout = old_out
        sys.stderr = old_err

    return _restore, fh


def main():
    parser = argparse.ArgumentParser(description='Detect people in images using YOLO11n')
    parser.add_argument('--model', default='models/people/yolo11n.pt', 
                       help='Path to YOLO model file for people detection')
    parser.add_argument('--input', default='inputs/samples', 
                       help='Input directory containing sample folders')
    parser.add_argument('--input_with_weapons', default=None, 
                       help='Input directory containing sample folders with weapons (optional)')
    parser.add_argument('--input_without_weapons', default=None, 
                       help='Input directory containing sample folders without weapons (optional)')
    parser.add_argument('--output', default='output/detections', 
                       help='Output directory for processed images')
    parser.add_argument('--person-confidence', type=float, default=0.5,
                       help='Confidence threshold for person detections')
    parser.add_argument('--weapon-confidence', type=float, default=0.5,
                       help='Confidence threshold for weapon detections')
    parser.add_argument('--sample-majority-threshold', type=int, default=5,
                       help='Number of frames with weapon detections needed to classify sample as having weapons')
    parser.add_argument('--save-crops', action='store_true', default=True,
                       help='Save individual person crops (default: True)')
    parser.add_argument('--no-crops', action='store_true',
                       help='Disable saving individual person crops')
    parser.add_argument('--no-weapons', action='store_true',
                       help='Disable weapon detection in person crops')
    parser.add_argument('--filter-clips', action='store_true',
                       help='Process only clips 0, 2, and 7 (clip_000, clip_002, clip_007)') # (0,45,-45)
    parser.add_argument('--dual-drone', action='store_true',
                       help='Enable dual-drone mode with two input directories')
    parser.add_argument('--input-drone1', default='inputs/samples/drone1',
                       help='Input directory for drone 1 (dual-drone mode)')
    parser.add_argument('--input-drone2', default='inputs/samples/drone2',
                       help='Input directory for drone 2 (dual-drone mode)')
    parser.add_argument('--association-threshold', type=float, default=100.0,
                       help='Distance threshold (meters) for associating detections across drones (default: 100.0)')

    parser.add_argument('--show-weapon-confidence', action='store_true',
                        help='Show weapon confidence text in overlays (default: False)')

    parser.add_argument('--log-level', default='INFO',
                        help='Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)')
    parser.add_argument('--log-dir', default='logs/',
                        help='Directory to write log files (default: logs/)')
    
    args = parser.parse_args()

    # Configure logging early so constructor logs are captured.
    log_level = _parse_log_level(args.log_level)
    log_dir = Path(args.log_dir) if args.log_dir else (Path(args.output) / "logs")

    # Also capture any print-based console output directly to a file.
    console_log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_console.log"
    restore_console, console_fh = _tee_console_to_file(console_log_path)
    
    try:
        # Handle crop saving logic
        save_crops = args.save_crops and not args.no_crops
        
        # Handle weapon detection logic
        enable_weapons = not args.no_weapons

        logging.getLogger(__name__).info(
            "Thresholds: PERSON_CONFIDENCE=%s WEAPON_CONFIDENCE=%s (weapons_enabled=%s)",
            args.person_confidence,
            args.weapon_confidence,
            enable_weapons,
        )
        
        # Validate inputs
        if not os.path.exists(args.model):
            logging.getLogger(__name__).error("Model file not found: %s", args.model)
            return 1
    
        # Determine input source
        input_dir = args.input
    
        # If specific weapon/no-weapon directories are provided, use them
        if args.input_with_weapons and os.path.exists(args.input_with_weapons):
            input_dir = args.input_with_weapons
        elif args.input_without_weapons and os.path.exists(args.input_without_weapons):
            input_dir = args.input_without_weapons
        elif not os.path.exists(input_dir):
            logging.getLogger(__name__).error("Input directory not found: %s", input_dir)
            return 1
    
        
        # Create output directory
        Path(args.output).mkdir(parents=True, exist_ok=True)
    
        # Check for dual-drone mode
        if args.dual_drone:
            if not args.input_drone1 or not args.input_drone2:
                logging.getLogger(__name__).error("Dual-drone mode requires both --input-drone1 and --input-drone2")
                return 1
            
            if not os.path.exists(args.input_drone1):
                logging.getLogger(__name__).error("Drone 1 input directory not found: %s", args.input_drone1)
                return 1
            if not os.path.exists(args.input_drone2):
                logging.getLogger(__name__).error("Drone 2 input directory not found: %s", args.input_drone2)
                return 1
            
            # Initialize dual-drone pipeline
            pipeline = DualDroneDetectionPipeline(
                args.model, 
                person_confidence_threshold=args.person_confidence,
                enable_weapon_detection=enable_weapons,
                weapon_confidence_threshold=args.weapon_confidence,
                sample_majority_threshold=args.sample_majority_threshold,
                association_threshold=args.association_threshold
            )
            
            # Set crop saving preference
            pipeline.save_crops = save_crops
            pipeline.show_weapon_confidence = bool(args.show_weapon_confidence)
            pipeline.pipeline_drone1.show_weapon_confidence = bool(args.show_weapon_confidence)
            pipeline.pipeline_drone2.show_weapon_confidence = bool(args.show_weapon_confidence)
            
            # Process dual-drone samples
            pipeline.process_dual_drone_samples(
                args.input_drone1,
                args.input_drone2,
                args.output,
                #filter_clips=args.filter_clips
            )

            # Print statistics summary
            print("\n=== DUAL-DRONE SUMMARY ===")
            print("\n--- Drone 1 ---")
            pipeline.stats_drone1.print_summary()
            print("\n--- Drone 2 ---")
            pipeline.stats_drone2.print_summary()
            print("\n--- Fused ---")
            pipeline.stats_fused.print_summary()
            
            return 0
    
        # Initialize single-drone pipeline
        pipeline = DetectionPipeline(
            args.model,
            person_confidence_threshold=args.person_confidence,
            enable_weapon_detection=enable_weapons,
            weapon_confidence_threshold=args.weapon_confidence,
            sample_majority_threshold=args.sample_majority_threshold,
        )
        
        # Set crop saving preference
        pipeline.save_crops = save_crops
        pipeline.show_weapon_confidence = bool(args.show_weapon_confidence)
        
        # Process all sample directories
        # Check if input is a single directory with images or a parent directory with subdirectories
        if os.path.isdir(input_dir):
            # Check if it's a single directory with images or contains subdirectories
            image_files = [f for f in os.listdir(input_dir) 
                          if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            if image_files:
                # Direct directory with images
                pipeline.process_directory(input_dir, args.output)
            else:
                # Directory with subdirectories
                pipeline.process_all_sample_directories(input_dir, args.output, filter_clips=args.filter_clips)
        else:
            logging.getLogger(__name__).error("Input path does not exist: %s", input_dir)
            return 1

        # Print comprehensive statistics
        pipeline.stats.print_summary()
        return 0
    finally:
        try:
            restore_console()
        finally:
            console_fh.close()


if __name__ == "__main__":
    exit(main())
