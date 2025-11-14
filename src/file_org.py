import os
import shutil
import glob

def organize_files(source_dir, data_dir):
    cropped_dir = os.path.join(data_dir, "cropped")
    binary_dir = os.path.join(data_dir, "binary")
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)

    # Move all *.cropped.tif files
    for cropped_file in glob.glob(os.path.join(source_dir, "*.cropped.tif")):
        dest = os.path.join(cropped_dir, os.path.basename(cropped_file))
        shutil.move(cropped_file, dest)
        print(f"Moved {cropped_file} -> {dest}")

    # Move all *.binary.tif files
    for binary_file in glob.glob(os.path.join(source_dir, "*.binary.tif")):
        dest = os.path.join(binary_dir, os.path.basename(binary_file))
        shutil.move(binary_file, dest)
        print(f"Moved {binary_file} -> {dest}")

if __name__ == "__main__":
    # Set your source and data directories here
    source_directory = "/home/bhunn1/vision_analysis/Ter94WT_GFP_SVIP_Lysotracker_Larvae1.lif"
    data_directory = "/home/bhunn1/vision_analysis/data"
    organize_files(source_directory, data_directory)