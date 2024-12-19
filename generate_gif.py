from PIL import Image
import os
import glob
import re

def natural_sort(item):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", item)]

folder_path = "output_masks/"

png_files = sorted(glob.glob(os.path.join(folder_path, "*.png")), key=natural_sort)
print(png_files)

if not png_files:
    print("No PNG files.")
    exit()

frames = [Image.open(png) for png in png_files]

output_path = os.path.join(folder_path, "output.gif")
frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=50, loop=0)

print(f"GIF created successfully at {output_path}")