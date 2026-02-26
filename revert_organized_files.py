import os
import shutil

def revert_organized_files(input_dir, output_dir):
    """
    Reverts organized files back to their original folder structure, considering only the first two folder levels.

    Args:
        input_dir (str): Path to the directory containing organized files.
        output_dir (str): Path to the directory where files will be reverted.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        # Skip non-MP4 and non-TXT files
        if not (file.endswith(".MP4") or file.endswith(".txt")):
            continue

        # Extract the original folder structure from the filename (first two levels)
        parts = file.split("_")
        if len(parts) < 3:  # Ensure there are at least two folder levels and a filename
            continue

        folder_path = os.path.join(output_dir, parts[0], parts[1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Move the file back to its original folder
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(folder_path, "_".join(parts[2:]))
        shutil.move(src_path, dest_path)

if __name__ == "__main__":
    input_directory = "inputs/organized_raw"
    output_directory = "inputs/raw_reverted"
    revert_organized_files(input_directory, output_directory)
