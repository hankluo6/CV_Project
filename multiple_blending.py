import cv2
import numpy as np
from PIL import Image
import utils
from seamless_cloning import PoissonSeamlessCloner
import os

def generate_contrail(contrail_unit_path, num_units=10, diminish_factor=0.8):
    # Load the contrail unit image
    contrail_unit = cv2.imread(contrail_unit_path, cv2.IMREAD_UNCHANGED)
    h, w = contrail_unit.shape[:2]
    
    contrails = []
    for i in range(num_units):
        # Diminish size
        scale = diminish_factor ** i
        dim = (int(w * scale), int(h * scale))
        resized = cv2.resize(contrail_unit, dim, interpolation=cv2.INTER_AREA)
        
        # Diminish opacity (for RGBA images)
        if resized.shape[2] == 4:  # Check if it has an alpha channel
            alpha = resized[:, :, 3] * (diminish_factor ** i)  # Reduce alpha
            resized[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
        
        contrails.append(resized)
        print("resized.shape: ", resized.shape)
        print("resized: ", resized)
    
    return contrails


def add_contrails(result, contrails, start_point, step=(10, 0)):
    x, y = start_point
    for contrail in contrails:
        h, w = contrail.shape[:2]

        if y + h > result.shape[0] or x + w > result.shape[1]:
            break

        alpha = contrail[:, :, 3] / 255.0  
        for c in range(3):
            result[y:y+h, x:x+w, c] = (result[y:y+h, x:x+w, c] * (1 - alpha) +contrail[:, :, c] * alpha)
        
        x += step[0]
        y += step[1]

    return result

def crop_image(input_image, crop_size, hop_size):
    h, w = input_image.shape[:2]
    crops = []
    crops_idx = []
    # for i in range(0, h, hop_size):
    for j in range(0, w, hop_size):
        crop = input_image[1000:1000+crop_size[0], j:j+crop_size[1]]
        if crop.shape[:2] == crop_size:
            crops.append(crop)
            crops_idx.append((1000, 1000+crop_size[0], j, j+crop_size[1]))
    return crops, crops_idx

def main(mask_path, source_image_path, target_image_path, contrail_unit_image_path, output_dir, solver='spsolve', scale=1.0, gradient_mixing_mode='max', gradient_mixing_alpha=1.0):
    # Read input and source images
    mask = utils.read_image(mask_path, scale=1.0, gray=True)
    src_rgb = utils.read_image(source_image_path, scale=1.0, gray=False)
    target_rgb = utils.read_image(target_image_path, scale=1.0,  gray=False)
    
    # Get the size of the source image
    source_size = src_rgb.shape[:2]
    
    # Crop the input image
    cropped_images, crop_idx = crop_image(target_rgb, source_size, 40)
    # Process each cropped image
    # back_ground = cv2.imread(target_image_path)
    for idx, target_image in enumerate(cropped_images):
        # Create a Poisson seamless cloner
        cloner = PoissonSeamlessCloner(mask, src_rgb, target_image, solver, 1.0)
        img = cloner.poisson_blend_rgb("max", 1.0)
        result = target_rgb.copy()
        result[crop_idx[idx][0]:crop_idx[idx][1], crop_idx[idx][2]:crop_idx[idx][3]] = img
        start_point = (crop_idx[idx][2] + 100, crop_idx[idx][0] + 50)
        result = add_contrails(result, contrails, start_point, step=(10, 0))
        result = (result * 255).astype(np.uint8)
        Image.fromarray(result).save(os.path.join(output_dir, "output_{}.png".format(idx)))
        print("Output image is saved as output_{}.png".format(idx))
        

if __name__ == "__main__":
    mask_path = "mask.jpg"
    source_image_path = "source.jpg"
    target_image_path = "target.jpg"
    output_dir = "output1"
    contrail_unit_image_path = "unit_contrail.png"
    # print("Contrails: ", len(contrails))
    
    main(mask_path, source_image_path, target_image_path, contrail_unit_image_path, output_dir)