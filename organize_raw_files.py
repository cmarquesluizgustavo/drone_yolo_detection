import os
import shutil

def organize_raw_files(input_dir, output_dir):
    """
    Organizes raw files by moving them into a single folder and renaming them to include their first two folder levels.

    Args:
        input_dir (str): Path to the input directory containing raw files.
        output_dir (str): Path to the output directory where organized files will be stored.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        # Extract the relative path from the input directory
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == ".":
            continue

        # Limit to the first two folder levels
        folder_levels = relative_path.split(os.sep)[:2]
        folder_name = "_".join(folder_levels)

        for file in files:
            # Skip non-MP4 and non-TXT files
            if not (file.endswith(".MP4") or file.endswith(".txt")):
                continue

            # Construct the new filename
            new_filename = f"{folder_name}_{file}"

            # Move the file to the output directory with the new name
            src_path = os.path.join(root, file)
            dest_path = os.path.join(output_dir, new_filename)
            shutil.move(src_path, dest_path)

if __name__ == "__main__":
    input_directory = "inputs/raw_reverted"
    output_directory = "inputs/organized_raw"
    organize_raw_files(input_directory, output_directory)
