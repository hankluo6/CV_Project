import cv2
import numpy as np
from PIL import Image
import utils
from seamless_cloning import PoissonSeamlessCloner
import os

def crop_image(input_image, crop_size, hop_size_x, y):
    h, w = input_image.shape[:2]
    crops = []
    crops_idx = []
    
    for j in range(0, w, hop_size_x):
        crop = input_image[y:y+crop_size[0], j:j+crop_size[1]]
        if crop.shape[:2] == crop_size:
            crops.append(crop)
            crops_idx.append((y, y+crop_size[0], j, j+crop_size[1]))
    return crops, crops_idx

def main(mask_path, source_image_path, target_image_path, output_dir, solver='spsolve', scale=1.0, gradient_mixing_mode='max', gradient_mixing_alpha=1.0):
    mask = utils.read_image(mask_path, scale=1.0, gray=True)
    src_rgb = utils.read_image(source_image_path, scale=1.0, gray=False)
    target_rgb = utils.read_image(target_image_path, scale=1.0,  gray=False)
    
    src_size = src_rgb.shape[:2]
    
    # Crop the input image
    cropped_images, crop_idx = crop_image(target_rgb, source_size, 40, 1000)
    # Process each cropped image
    # back_ground = cv2.imread(target_image_path)
    for idx, target_image in enumerate(cropped_images):
        cloner = PoissonSeamlessCloner(mask, src_rgb, target_image, solver, 1.0)
        img = cloner.poisson_blend_rgb("max", 1.0)
        result = target_rgb.copy()
        result[crop_idx[idx][0]:crop_idx[idx][1], crop_idx[idx][2]:crop_idx[idx][3]] = img
        result = (result * 255).astype(np.uint8)
        Image.fromarray(result).save(os.path.join(output_dir, "output_{}.png".format(idx)))
        print("Output image is saved as output_{}.png".format(idx))
        

if __name__ == "__main__":
    mask_path = "source.png"
    source_image_path = "source.jpg"
    target_image_path = "target.jpg"
    output_dir = "output"
    
    main(mask_path, source_image_path, target_image_path, output_dir)