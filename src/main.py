#!/usr/bin/env python3
"""
Main script to run people detection pipeline.
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection_pipeline import DetectionPipeline


def main():
    parser = argparse.ArgumentParser(description='Detect people in images using YOLOv11')
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
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--weapon-confidence', type=float, default=0.2,
                       help='Confidence threshold for weapon detections (default: 0.2)')
    parser.add_argument('--sample-majority-threshold', type=int, default=1,
                       help='Number of frames with weapon detections needed to classify sample as having weapons (default: 1)')
    parser.add_argument('--save-crops', action='store_true', default=True,
                       help='Save individual person crops (default: True)')
    parser.add_argument('--no-crops', action='store_true',
                       help='Disable saving individual person crops')
    parser.add_argument('--no-weapons', action='store_true',
                       help='Disable weapon detection in person crops')
    parser.add_argument('--filter-clips', action='store_true',
                       help='Process only clips 0, 2, and 7 (clip_000, clip_002, clip_007)')
    
    args = parser.parse_args()
    
    # Handle crop saving logic
    save_crops = args.save_crops and not args.no_crops
    
    # Handle weapon detection logic
    enable_weapons = not args.no_weapons
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Determine input source
    input_dir = args.input
    
    # If specific weapon/no-weapon directories are provided, use them
    if args.input_with_weapons and os.path.exists(args.input_with_weapons):
        input_dir = args.input_with_weapons
    elif args.input_without_weapons and os.path.exists(args.input_without_weapons):
        input_dir = args.input_without_weapons
    elif not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    print(f"Initializing detection pipeline with model: {args.model}")
    pipeline = DetectionPipeline(args.model, args.confidence, enable_weapon_detection=enable_weapons, weapon_confidence_threshold=args.weapon_confidence, sample_majority_threshold=args.sample_majority_threshold)
    
    # Set crop saving preference
    pipeline.save_crops = save_crops
    
    # Process all sample directories
    print(f"Processing samples from: {input_dir}")
    print(f"Output will be saved to: {args.output}")
    print(f"Crop saving: {'Enabled' if save_crops else 'Disabled'}")
    print(f"Weapon detection: {'Enabled' if enable_weapons and pipeline.enable_weapon_detection else 'Disabled'}")
    if enable_weapons and pipeline.enable_weapon_detection:
        print(f"Weapon confidence threshold: {args.weapon_confidence}")
    print(f"Sample majority threshold: {args.sample_majority_threshold} frame(s)")
    print(f"Ground truth determined from filenames: 'real*' = has weapons, 'falso*' = no weapons")
    if args.filter_clips:
        print(f"Filtering to clips: 0, 2, 7 only")
    
    # Check if input is a single directory with images or a parent directory with subdirectories
    if os.path.isdir(input_dir):
        # Check if it's a single directory with images or contains subdirectories
        image_files = [f for f in os.listdir(input_dir) 
                      if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if image_files:
            # Direct directory with images
            print(f"Processing single directory with {len(image_files)} images")
            pipeline.process_directory(input_dir, args.output)
        else:
            # Directory with subdirectories
            pipeline.process_all_sample_directories(input_dir, args.output, filter_clips=args.filter_clips)
    else:
        print(f"Error: Input path does not exist: {input_dir}")
        return 1

    # Print comprehensive statistics
    pipeline.stats.print_summary()
    print("\nProcessing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
