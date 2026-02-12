import os
import shutil

def collect_frame_0003_to_view(base_detections_dir, base_view_dir):
    for angle in ["45", "90"]:
        angle_dir = os.path.join(base_detections_dir, f"angle_{angle}", "detections_dual_drone")
        for drone in ["drone1", "drone2"]:
            dest_dir = os.path.join(base_view_dir, angle, drone)
            os.makedirs(dest_dir, exist_ok=True)
            count = 0
            for subfolder in os.listdir(angle_dir):
                if subfolder.endswith(drone):
                    subfolder_path = os.path.join(angle_dir, subfolder)
                    if os.path.isdir(subfolder_path):
                        for file in os.listdir(subfolder_path):
                            if file.startswith("frame_0003"):
                                src = os.path.join(subfolder_path, file)
                                ext = os.path.splitext(file)[1]
                                dest_file = f"{subfolder}_frame_0003{ext}"
                                dest = os.path.join(dest_dir, dest_file)
                                shutil.copy2(src, dest)
                                count += 1
            print(f"Copied {count} files to {dest_dir}")

if __name__ == "__main__":
    base_detections_dir = r"/home/caio.torkst/Downloads/drone_yolo_detection/output/detections"
    base_view_dir = os.path.join(base_detections_dir, "view")
    collect_frame_0003_to_view(base_detections_dir, base_view_dir)