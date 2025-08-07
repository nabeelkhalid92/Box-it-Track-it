#!/usr/bin/env python3
"""
convert_tiff_to_jpg.py

Usage:
    python convert_tiff_to_jpg.py /path/to/tiff_directory

This script will:
1. Find all .tiff files in the specified directory.
2. Create a sub-directory named 'converted_to_jpg' (if it doesn't exist).
3. Convert each .tiff file to .jpg format and save in that sub-directory.

Make sure you have 'Pillow' installed:
    pip install Pillow
"""

import sys
import os
from PIL import Image

def convert_tiff_to_jpg(input_dir):
    # Create the output directory inside the input directory
    output_dir = os.path.join(input_dir, "converted_to_jpg")
    os.makedirs(output_dir, exist_ok=True)

    # Count how many files were converted
    count = 0

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        # Check if file is a .tiff (case-insensitive)
        if filename.lower().endswith(".tiff"):
            file_path = os.path.join(input_dir, filename)
            base_name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{base_name}.jpg")

            try:
                with Image.open(file_path) as img:
                    # Convert image to RGB and save as JPEG
                    img.convert("RGB").save(output_path, "JPEG")
                print(f"Converted: {file_path} --> {output_path}")
                count += 1
            except Exception as e:
                print(f"Error converting file {file_path}: {e}")

    print(f"\nTotal .tiff files converted to .jpg: {count}")
    print(f"Converted files are located in: {output_dir}")

def main():
    # Check if the user passed a directory argument
    if len(sys.argv) != 2:
        print("Usage: python convert_tiff_to_jpg.py /path/to/tiff_directory")
        sys.exit(1)

    tiff_directory = sys.argv[1]

    # Check if the provided path is a directory
    if not os.path.isdir(tiff_directory):
        print(f"Error: {tiff_directory} is not a valid directory.")
        sys.exit(1)

    convert_tiff_to_jpg(tiff_directory)

if __name__ == "__main__":
    main()
