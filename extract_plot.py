import os
import shutil

def move_plots_out_of_subfolders(base_dir):
    for angle in ['45', '90']:
        angle_dir = os.path.join(base_dir, angle)
        if not os.path.isdir(angle_dir):
            continue
        for subfolder in os.listdir(angle_dir):
            subfolder_path = os.path.join(angle_dir, subfolder)
            if os.path.isdir(subfolder_path):
                # Determine drone from subfolder name if possible
                drone = None
                if 'drone1' in subfolder:
                    drone = 'drone1'
                elif 'drone2' in subfolder:
                    drone = 'drone2'
                # If drone is found, move to plots/{angle}/{drone}/
                if drone:
                    dest_dir = os.path.join(angle_dir, drone)
                    os.makedirs(dest_dir, exist_ok=True)
                else:
                    dest_dir = angle_dir
                for filename in os.listdir(subfolder_path):
                    src = os.path.join(subfolder_path, filename)
                    dst = os.path.join(dest_dir, filename)
                    # Avoid overwriting files with the same name
                    if os.path.exists(dst):
                        print(f"File {dst} already exists, skipping.")
                        continue
                    shutil.move(src, dst)
                # Remove the empty subfolder
                os.rmdir(subfolder_path)

if __name__ == "__main__":
    base_plots_dir = r"d:\DRONE\drone_yolo_detection\output\detections\plots"
    move_plots_out_of_subfolders(base_plots_dir)
    print("Done.")