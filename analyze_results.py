#!/usr/bin/env python3
"""
Analysis script for dual-drone detection results
Processes detection outputs and generates metrics summaries
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_log_metrics(log_path):
    """Parse metrics from console log file"""
    metrics = {
        'angle_45': {},
        'angle_90': {}
    }
    
    current_angle = None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Detect angle section
        if "PROCESSING ANGLE: 45°" in line:
            current_angle = 'angle_45'
        elif "PROCESSING ANGLE: 90°" in line:
            current_angle = 'angle_90'
        
        # Parse final metrics
        if "OVERALL PER-FRAME METRICS:" in line and current_angle:
            # Read next lines for overall metrics
            for j in range(i+1, min(i+10, len(lines))):
                if "Accuracy:" in lines[j]:
                    metrics[current_angle]['overall_accuracy'] = float(lines[j].split(':')[1].strip())
                elif "Precision:" in lines[j]:
                    metrics[current_angle]['overall_precision'] = float(lines[j].split(':')[1].strip())
                elif "Recall:" in lines[j]:
                    metrics[current_angle]['overall_recall'] = float(lines[j].split(':')[1].strip())
                elif "F1-Score:" in lines[j]:
                    metrics[current_angle]['overall_f1'] = float(lines[j].split(':')[1].strip())
                elif "TP:" in lines[j]:
                    parts = lines[j].split(',')
                    metrics[current_angle]['TP'] = int(parts[0].split(':')[1].strip())
                    metrics[current_angle]['TN'] = int(parts[1].split(':')[1].strip())
                    metrics[current_angle]['FP'] = int(parts[2].split(':')[1].strip())
                    metrics[current_angle]['FN'] = int(parts[3].split(':')[1].strip())
        
        # Parse per-sample metrics
        if "PER-SAMPLE METRICS (Majority Threshold:" in line and current_angle:
            for j in range(i+1, min(i+10, len(lines))):
                if "Accuracy:" in lines[j]:
                    metrics[current_angle]['sample_accuracy'] = float(lines[j].split(':')[1].strip())
                elif "Precision:" in lines[j]:
                    metrics[current_angle]['sample_precision'] = float(lines[j].split(':')[1].strip())
                elif "Recall:" in lines[j]:
                    metrics[current_angle]['sample_recall'] = float(lines[j].split(':')[1].strip())
                elif "F1-Score:" in lines[j]:
                    metrics[current_angle]['sample_f1'] = float(lines[j].split(':')[1].strip())
                    break
        
        # Parse RMSE
        if "FUSED-GEO DISTANCE RMSE:" in line and current_angle:
            rmse_str = line.split(':')[1].strip()
            metrics[current_angle]['rmse'] = float(rmse_str.split('m')[0].strip())
            measurements = rmse_str.split('(')[1].split()[0]
            metrics[current_angle]['rmse_measurements'] = int(measurements)
    
    return metrics

def analyze_detections(output_dir):
    """Analyze detection JSON files"""
    results = defaultdict(lambda: defaultdict(list))
    
    for angle in ['angle_45', 'angle_90']:
        fused_dir = Path(output_dir) / angle / 'fused_detections'
        
        if not fused_dir.exists():
            continue
        
        for sample_dir in fused_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            
            sample_name = sample_dir.name
            sample_type = 'real' if 'real' in sample_name else 'falso'
            
            # Distance info from sample name
            parts = sample_name.split('_')
            drone_distance = parts[1]  # e.g., '05' or '10'
            camera_height = parts[2]   # e.g., '02' or '05'
            
            # Look for JSON files
            for json_file in sample_dir.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract key metrics
                    if 'detections' in data and len(data['detections']) > 0:
                        detection = data['detections'][0]
                        
                        results[angle]['has_detection'].append(len(data['detections']) > 0)
                        results[angle]['confidence'].append(detection.get('confidence', 0))
                        results[angle]['has_weapon'].append(detection.get('has_weapon', False))
                        results[angle]['weapon_confidence'].append(detection.get('weapon_confidence', 0))
                        results[angle]['sample_type'].append(sample_type)
                        results[angle]['drone_distance'].append(drone_distance)
                        results[angle]['camera_height'].append(camera_height)
                        
                        # Distance estimates
                        if 'estimated_distance_m' in detection:
                            results[angle]['distance_estimate'].append(detection['estimated_distance_m'])
                
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
    
    return results

def create_summary_report(metrics, detection_results):
    """Create a comprehensive summary report"""
    
    print("\n" + "="*80)
    print(" DUAL-DRONE WEAPON DETECTION - ANALYSIS REPORT")
    print("="*80 + "\n")
    
    # Overall metrics comparison
    print("📊 OVERALL METRICS COMPARISON")
    print("-" * 80)
    
    comparison_data = []
    for angle in ['angle_45', 'angle_90']:
        if metrics[angle]:
            comparison_data.append({
                'Angle': angle.replace('angle_', '') + '°',
                'Accuracy': f"{metrics[angle].get('overall_accuracy', 0):.3f}",
                'Precision': f"{metrics[angle].get('overall_precision', 0):.3f}",
                'Recall': f"{metrics[angle].get('overall_recall', 0):.3f}",
                'F1-Score': f"{metrics[angle].get('overall_f1', 0):.3f}",
                'RMSE (m)': f"{metrics[angle].get('rmse', 0):.2f}"
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
    
    # Detailed metrics per angle
    for angle in ['angle_45', 'angle_90']:
        if not metrics[angle]:
            continue
        
        angle_name = angle.replace('angle_', '') + '°'
        print(f"\n\n{'='*80}")
        print(f" ANGLE {angle_name} - DETAILED METRICS")
        print(f"{'='*80}\n")
        
        # Per-frame metrics
        print("🎯 PER-FRAME METRICS (Weapon Detection)")
        print("-" * 80)
        m = metrics[angle]
        
        print(f"  Accuracy:  {m.get('overall_accuracy', 0):.3f}")
        print(f"  Precision: {m.get('overall_precision', 0):.3f}")
        print(f"  Recall:    {m.get('overall_recall', 0):.3f}")
        print(f"  F1-Score:  {m.get('overall_f1', 0):.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {m.get('TP', 0):<4} TN: {m.get('TN', 0):<4}")
        print(f"    FP: {m.get('FP', 0):<4} FN: {m.get('FN', 0):<4}")
        
        # Per-sample metrics
        print(f"\n\n📦 PER-SAMPLE METRICS (Video Clip Level)")
        print("-" * 80)
        print(f"  Accuracy:  {m.get('sample_accuracy', 0):.3f}")
        print(f"  Precision: {m.get('sample_precision', 0):.3f}")
        print(f"  Recall:    {m.get('sample_recall', 0):.3f}")
        print(f"  F1-Score:  {m.get('sample_f1', 0):.3f}")
        
        # Distance estimation
        print(f"\n\n📏 DISTANCE ESTIMATION")
        print("-" * 80)
        print(f"  RMSE: {m.get('rmse', 0):.2f} meters")
        print(f"  Measurements: {m.get('rmse_measurements', 0)}")
        
        # Detection statistics
        if angle in detection_results and detection_results[angle]:
            print(f"\n\n📈 DETECTION STATISTICS")
            print("-" * 80)
            
            dr = detection_results[angle]
            if 'confidence' in dr and dr['confidence']:
                print(f"  Average Person Confidence: {np.mean(dr['confidence']):.3f}")
                print(f"  Min/Max Confidence: {np.min(dr['confidence']):.3f} / {np.max(dr['confidence']):.3f}")
            
            if 'weapon_confidence' in dr and dr['weapon_confidence']:
                weapon_confs = [w for w in dr['weapon_confidence'] if w > 0]
                if weapon_confs:
                    print(f"  Average Weapon Confidence: {np.mean(weapon_confs):.3f}")
                    print(f"  Weapon Detections: {len(weapon_confs)} / {len(dr['weapon_confidence'])}")
            
            # By distance and height
            if 'drone_distance' in dr and dr['drone_distance']:
                print(f"\n  By Drone Distance:")
                for dist in sorted(set(dr['drone_distance'])):
                    count = dr['drone_distance'].count(dist)
                    print(f"    {dist}m: {count} frames")
            
            if 'camera_height' in dr and dr['camera_height']:
                print(f"\n  By Camera Height:")
                for height in sorted(set(dr['camera_height'])):
                    count = dr['camera_height'].count(height)
                    print(f"    {height}m: {count} frames")
    
    # Key findings
    print(f"\n\n{'='*80}")
    print(" KEY FINDINGS")
    print(f"{'='*80}\n")
    
    if metrics['angle_45'] and metrics['angle_90']:
        f1_45 = metrics['angle_45'].get('sample_f1', 0)
        f1_90 = metrics['angle_90'].get('sample_f1', 0)
        
        if f1_45 > f1_90:
            best_angle = "45°"
            diff = (f1_45 - f1_90) * 100
        else:
            best_angle = "90°"
            diff = (f1_90 - f1_45) * 100
        
        print(f"✓ Best performing angle: {best_angle} (F1-Score difference: {diff:.1f}%)")
        
        rmse_45 = metrics['angle_45'].get('rmse', 0)
        rmse_90 = metrics['angle_90'].get('rmse', 0)
        
        if rmse_45 < rmse_90:
            better_dist = "45°"
        else:
            better_dist = "90°"
        
        print(f"✓ Better distance estimation: {better_dist}")
        print(f"  - 45° RMSE: {rmse_45:.2f}m")
        print(f"  - 90° RMSE: {rmse_90:.2f}m")
    
    print("\n" + "="*80 + "\n")

def create_detailed_tables(metrics, detection_results):
    """Create detailed comparison tables"""
    
    print("\n" + "="*100)
    print(" COMPREHENSIVE METRICS TABLES")
    print("="*100 + "\n")
    
    # Table 1: Per-Frame Metrics Comparison
    print("📊 TABLE 1: PER-FRAME WEAPON DETECTION METRICS")
    print("-" * 100)
    
    frame_data = []
    for angle in ['angle_45', 'angle_90']:
        if metrics[angle]:
            m = metrics[angle]
            frame_data.append({
                'Angle': angle.replace('angle_', '') + '°',
                'Accuracy': f"{m.get('overall_accuracy', 0):.4f}",
                'Precision': f"{m.get('overall_precision', 0):.4f}",
                'Recall': f"{m.get('overall_recall', 0):.4f}",
                'F1-Score': f"{m.get('overall_f1', 0):.4f}",
                'TP': m.get('TP', 0),
                'TN': m.get('TN', 0),
                'FP': m.get('FP', 0),
                'FN': m.get('FN', 0)
            })
    
    if frame_data:
        df = pd.DataFrame(frame_data)
        print(df.to_string(index=False))
        print()
    
    # Table 2: Per-Sample Metrics Comparison
    print("\n📦 TABLE 2: PER-SAMPLE (VIDEO CLIP) METRICS")
    print("-" * 100)
    
    sample_data = []
    for angle in ['angle_45', 'angle_90']:
        if metrics[angle]:
            m = metrics[angle]
            sample_data.append({
                'Angle': angle.replace('angle_', '') + '°',
                'Accuracy': f"{m.get('sample_accuracy', 0):.4f}",
                'Precision': f"{m.get('sample_precision', 0):.4f}",
                'Recall': f"{m.get('sample_recall', 0):.4f}",
                'F1-Score': f"{m.get('sample_f1', 0):.4f}"
            })
    
    if sample_data:
        df = pd.DataFrame(sample_data)
        print(df.to_string(index=False))
        print()
    
    # Table 3: Distance Estimation Performance
    print("\n📏 TABLE 3: DISTANCE ESTIMATION PERFORMANCE")
    print("-" * 100)
    
    distance_data = []
    for angle in ['angle_45', 'angle_90']:
        if metrics[angle] and 'rmse' in metrics[angle]:
            m = metrics[angle]
            distance_data.append({
                'Angle': angle.replace('angle_', '') + '°',
                'RMSE (meters)': f"{m.get('rmse', 0):.2f}",
                'Measurements': m.get('rmse_measurements', 0),
                'Avg Error per Frame': f"{m.get('rmse', 0):.2f}m"
            })
    
    if distance_data:
        df = pd.DataFrame(distance_data)
        print(df.to_string(index=False))
        print()
    
    # Table 4: Confusion Matrices Side by Side
    print("\n🎯 TABLE 4: CONFUSION MATRICES")
    print("-" * 100)
    
    for angle in ['angle_45', 'angle_90']:
        if metrics[angle]:
            angle_name = angle.replace('angle_', '') + '°'
            m = metrics[angle]
            
            print(f"\n{angle_name} Angle:")
            print("                  Predicted")
            print("                  No Weapon    Has Weapon")
            print(f"Actual No Weapon     {m.get('TN', 0):>6}        {m.get('FP', 0):>6}")
            print(f"       Has Weapon    {m.get('FN', 0):>6}        {m.get('TP', 0):>6}")
    
    # Table 5: Detection Statistics by Distance and Height (if available)
    if detection_results:
        print("\n\n📈 TABLE 5: DETECTION STATISTICS BY CONFIGURATION")
        print("-" * 100)
        
        for angle in ['angle_45', 'angle_90']:
            if angle in detection_results and detection_results[angle]:
                dr = detection_results[angle]
                angle_name = angle.replace('angle_', '') + '°'
                
                print(f"\n{angle_name} Angle:")
                
                # By drone distance
                if 'drone_distance' in dr and dr['drone_distance']:
                    print("\n  By Drone Distance:")
                    dist_stats = {}
                    for i, dist in enumerate(dr['drone_distance']):
                        if dist not in dist_stats:
                            dist_stats[dist] = {'count': 0, 'confidences': []}
                        dist_stats[dist]['count'] += 1
                        if i < len(dr['confidence']):
                            dist_stats[dist]['confidences'].append(dr['confidence'][i])
                    
                    dist_table = []
                    for dist in sorted(dist_stats.keys()):
                        avg_conf = np.mean(dist_stats[dist]['confidences']) if dist_stats[dist]['confidences'] else 0
                        dist_table.append({
                            'Distance': f"{dist}m",
                            'Frames': dist_stats[dist]['count'],
                            'Avg Confidence': f"{avg_conf:.3f}"
                        })
                    
                    if dist_table:
                        df = pd.DataFrame(dist_table)
                        print(df.to_string(index=False))
                
                # By camera height
                if 'camera_height' in dr and dr['camera_height']:
                    print("\n  By Camera Height:")
                    height_stats = {}
                    for i, height in enumerate(dr['camera_height']):
                        if height not in height_stats:
                            height_stats[height] = {'count': 0, 'confidences': []}
                        height_stats[height]['count'] += 1
                        if i < len(dr['confidence']):
                            height_stats[height]['confidences'].append(dr['confidence'][i])
                    
                    height_table = []
                    for height in sorted(height_stats.keys()):
                        avg_conf = np.mean(height_stats[height]['confidences']) if height_stats[height]['confidences'] else 0
                        height_table.append({
                            'Height': f"{height}m",
                            'Frames': height_stats[height]['count'],
                            'Avg Confidence': f"{avg_conf:.3f}"
                        })
                    
                    if height_table:
                        df = pd.DataFrame(height_table)
                        print(df.to_string(index=False))
    
    print("\n" + "="*100)

def main():
    # Paths
    base_dir = Path(__file__).parent
    log_path = base_dir / 'logs' / 'run_20260203_104748_console.log'
    output_dir = base_dir / 'output' / 'detections'
    
    print("\n🔍 Analyzing dual-drone detection results...")
    
    # Parse metrics from log
    print("  → Parsing console log...")
    metrics = parse_log_metrics(log_path)
    
    # Analyze detection files
    print("  → Analyzing detection files...")
    detection_results = analyze_detections(output_dir)
    
    # Generate report
    create_summary_report(metrics, detection_results)
    
    # Create detailed tables
    print("\n� Creating detailed comparison tables...")
    create_detailed_tables(metrics, detection_results)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
