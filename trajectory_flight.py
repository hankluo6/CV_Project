import cv2
import numpy as np
from PIL import Image
import utils
from seamless_cloning import PoissonSeamlessCloner
import os
from extract_ridge import extract_ridge_line
from rotate_image import rotate_image
from scipy.ndimage import gaussian_filter1d

def get_smoothed_derivative(spl, x, smooth_neighbors=100, sigma=250.0):
    dy_dx = spl.derivative()
    x_smooth = np.linspace(x - smooth_neighbors // 2, x + smooth_neighbors // 2, smooth_neighbors)
    smoothed_derivative = gaussian_filter1d(dy_dx(x_smooth), sigma=sigma)
    return smoothed_derivative[smooth_neighbors // 2]
    
def crop_image(input_image, crop_size, hop_size, spl):
    _, w = input_image.shape[:2]
    crops = []
    crops_idx = []
    angles = []
    spl_derivative = spl.derivative()
    for j in range(0, w, hop_size):
        mid_x = j + crop_size[1] // 2
        mid_y = int(spl(mid_x))
        y_start = max(0, mid_y - crop_size[0])

        crop = input_image[y_start:y_start+crop_size[0], j:j+crop_size[1]]
        if crop.shape[:2] == crop_size:
            crops.append(crop)
            crops_idx.append((y_start, y_start+crop_size[0], j, j+crop_size[1]))

            derivative = get_smoothed_derivative(spl, mid_x)
            angles.append(np.degrees(np.arctan(-derivative)).item())
    
    return crops, crops_idx, angles

def main(mask_path, source_image_path, target_image_path, output_dir, solver='spsolve', scale=1.0, gradient_mixing_mode='max', gradient_mixing_alpha=1.0):
    src_rgb = utils.read_image(source_image_path, scale=1.0, gray=False)
    target_rgb = utils.read_image(target_image_path, scale=5.0,  gray=False)
    src_size = src_rgb.shape[:2]

    spl = extract_ridge_line(target_image_path, scale=5.0)
    
    img_mask = cv2.imread(mask_path)
    img_src = cv2.imread(source_image_path)
    cropped_images, crop_idx, angles = crop_image(target_rgb, src_size, 5, spl)
    for idx, target_image in enumerate(cropped_images):
        mask = rotate_image(img_mask, angle=angles[idx], gray=True)
        src_rgb = rotate_image(img_src, angle=angles[idx], gray=False)

        cloner = PoissonSeamlessCloner(mask, src_rgb, target_image, solver, 1.0)
        img = cloner.poisson_blend_rgb("max", 1.0)
        result = target_rgb.copy()
        result[crop_idx[idx][0]:crop_idx[idx][1], crop_idx[idx][2]:crop_idx[idx][3]] = img
        result = (result * 255).astype(np.uint8)
        Image.fromarray(result).save(os.path.join(output_dir, "output_{}.png".format(idx)))
        print("Output image is saved as output_{}.png".format(idx))
        

if __name__ == "__main__":
    mask_path = "mask.jpg"
    source_image_path = "source.jpg"
    target_image_path = "mountain.jpg"
    output_dir = "output"
    
    main(mask_path, source_image_path, target_image_path, output_dir)