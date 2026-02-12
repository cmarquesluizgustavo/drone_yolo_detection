import os
import shutil

def move_ground_gps_plots():
    base_dir = "/home/caio.torkst/Downloads/drone_yolo_detection/output/detections"
    plots_base = "/home/caio.torkst/Downloads/drone_yolo_detection/output/plots"
    for angle in ["angle_45", "angle_90"]:
        angle_num = angle.split('_')[1]
        fused_dir = os.path.join(base_dir, angle, "fused_detections")
        if not os.path.isdir(fused_dir):
            continue
        dest_dir = os.path.join(plots_base, angle_num)
        os.makedirs(dest_dir, exist_ok=True)
        for subfolder in os.listdir(fused_dir):
            subfolder_path = os.path.join(fused_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            # Only process subfolders for clip_000_1080p
            if "clip_000_1080p" not in subfolder:
                continue
            for filename in os.listdir(subfolder_path):
                expected_prefix = subfolder
                expected_suffix = "_frame_0000_ground_gps.png"
                if filename.startswith(expected_prefix) and filename.endswith(expected_suffix):
                    src = os.path.join(subfolder_path, filename)

                    # Extract class, dist, and height from the filename
                    parts = filename.split('_')
                    if len(parts) < 5:
                        print(f"Filename {filename} does not match expected pattern, skipping.")
                        continue
                    class_name = parts[0]
                    dist = parts[1]
                    height = parts[2]

                    # Construct the new filename
                    new_filename = f"{class_name}_{dist}_{height}_ground_gps.png"
                    dst = os.path.join(dest_dir, new_filename)

                    if os.path.exists(dst):
                        print(f"File {dst} already exists, skipping.")
                        continue
                    shutil.copy2(src, dst)
                    print(f"Copied {src} -> {dst}")

if __name__ == "__main__":
    move_ground_gps_plots()
    print("Done.")