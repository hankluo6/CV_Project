import cv2
import numpy as np
from PIL import Image, ImageSequence
import utils
from seamless_cloning import PoissonSeamlessCloner
import os
import re

gif_length = 0
SIZE = (300, 225)

def crop_image(input_image, crop_size, hop_size_x, y):
    h, w = input_image.shape[:2]
    crops = []
    crops_idx = []
    for j in range(0, w, hop_size_x):
        crop = input_image[y : y + crop_size[0], j : j + crop_size[1]]
        if crop.shape[:2] == crop_size:
            crops.append(crop)
            crops_idx.append((y, y + crop_size[0], j, j + crop_size[1]))
    return crops, crops_idx

def load_gif(path):
    global gif_length
    result = []
    gif = Image.open(path)
    for i, frame in enumerate(ImageSequence.Iterator(gif)):
        # save
        frame = frame.convert('RGB')
        Image.fromarray(np.array(frame)).save(
            os.path.join('frames/', "frame_{}.png".format(i)))
        result.append(utils.read_image(os.path.join('frames/', 'frame_{}.png').format(i), scale=1.0, gray=False))
    gif_length = len(result)
    return result

def load_mask(path):
    masks = [utils.read_image(mask_path + "/" + mask, gray=True) for mask in sorted(os.listdir(mask_path), key=natural_sort_key)]
    return masks

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main(
    mask_path,
    source_image_path,
    target_image_path,
    output_dir,
    solver="spsolve",
    scale=1.0,
    gradient_mixing_mode="max",
    gradient_mixing_alpha=1.0,
):
    # Read input and source images
    # mask = utils.read_image(mask_path, scale=1.0, gray=True)
    # src_rgb = utils.read_image(source_image_path, scale=1.0, gray=False)
    src_imgs = load_gif(source_image_path)
    masks = load_mask(mask_path)
    for i in range(gif_length):
        src_imgs[i] = cv2.resize(src_imgs[i], SIZE)
        masks[i] = cv2.resize(masks[i], SIZE)
    target_rgb = utils.read_image(target_image_path, scale=1.0, gray=False)



    # Get the size of the source image
    source_size = src_imgs[0].shape[:2]

    # Crop the input image
    cropped_images, crop_idx = crop_image(target_rgb, source_size, 3, 600)
    # Process each cropped image
    # back_ground = cv2.imread(target_image_path)
    for idx, target_image in enumerate(cropped_images):
        # Create a Poisson seamless cloner
        gif_idx = idx % gif_length
        mask, src_rgb = masks[gif_idx], src_imgs[gif_idx]
        
        cloner = PoissonSeamlessCloner(mask, src_rgb, target_image, solver, 1.0)
        img = cloner.poisson_blend_rgb(gradient_mixing_mode, gradient_mixing_alpha)
        img_save = (img * 255).astype(np.uint8)

        # Image.fromarray(img_save).save(
        #     os.path.join('imgs/', "output_{}.png".format(idx))
        # )

        result = target_rgb.copy()
        result[
            crop_idx[idx][0] : crop_idx[idx][1], crop_idx[idx][2] : crop_idx[idx][3]
        ] = img
        result = (result * 255).astype(np.uint8)
        Image.fromarray(result).save(
            os.path.join(output_dir, "output_{}.png".format(idx))
        )
        print("Output image is saved as output_{}.png".format(idx))


if __name__ == "__main__":
    mask_path = "mask_origin"
    source_image_path = "walking_man.gif"
    target_image_path = "stage.png"
    output_dir = "red_output_origin"

    main(mask_path, source_image_path, target_image_path, output_dir)
