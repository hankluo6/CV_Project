import cv2
import numpy as np
from PIL import Image
import utils
from seamless_cloning import PoissonSeamlessCloner
import os
from edge_detect import detect_edges_and_smooth_curve
from rotate_image import rotate_and_resize_image
from scipy.ndimage import gaussian_filter1d, convolve1d

def compute_gradient(edge_curve, sigma=1.0):
    y = edge_curve[:, 1]
    x = edge_curve[:, 0]

    # Define a central difference kernel for higher-order accuracy
    # kernel = np.array([1, -8, 12, -15, 18, -24, 30, -36, 42, -48, 54, -60, 60, -54, 48, -42, 36, -30, 24, -18, 15, -12, 8, -1]) / 420
    kernel = np.array([1, -8, 0, 8, -1]) / 12

    # Compute the gradient using convolution
    dy_dx = convolve1d(y, kernel, mode='nearest') / (x[1] - x[0])

    return dy_dx

def find_pos_y(edge_curve, x):
    y = np.interp(x, edge_curve[:, 0], edge_curve[:, 1])
    return int(y)

def find_angle(edge_curve, x):
    # Find the nearest x in the edge_curve
    nearest_x_idx = np.abs(edge_curve[:, 0] - x).argmin()
    print(nearest_x_idx)
    angle = np.arctan(edge_curve[nearest_x_idx][2]) * 180 / np.pi
    return angle.item()

def get_smoothed_derivative(spl, x, smooth_nearby=100, sigma=250.0):
    # Generate points nearby x, use 20 points
    x_smooth = np.linspace(x - smooth_nearby // 2, x + smooth_nearby // 2, smooth_nearby)
    dy_dx = spl.derivative()
    # Compute the central gradient based on all nearby points
    slope_smoothed = gaussian_filter1d(-dy_dx(x_smooth), sigma=sigma)
    return slope_smoothed[smooth_nearby // 2]
    
# def crop_image(input_image, crop_size, hop_size, edge_curve):
def crop_image(input_image, crop_size, hop_size, spl):
    h, w = input_image.shape[:2]
    crops = []
    crops_idx = []
    angles = []
    spl_derivative = spl.derivative()
    for j in range(0, w, hop_size):
        mid_x = j + crop_size[1] // 2
        # mid_y = find_pos_y(edge_curve, mid_x)
        mid_y = int(spl(mid_x))
        y_start = max(0, mid_y - crop_size[0])
        crop = input_image[y_start:y_start+crop_size[0], j:j+crop_size[1]]
        if crop.shape[:2] == crop_size:
            crops.append(crop)
            crops_idx.append((y_start, y_start+crop_size[0], j, j+crop_size[1]))
        # angles.append(find_angle(edge_curve, mid_x))
        # angle = np.degrees(np.arctan(spl_derivative(mid_x)))
        # angles.append(angle.item())
        derivative = get_smoothed_derivative(spl, mid_x)
        angle = np.degrees(np.arctan(derivative))
        angles.append(angle.item())
        print('x: {}, y: {}, dy_dx: {}, angle: {}'.format(mid_x, mid_y, spl_derivative(mid_x), angle.item()))
    return crops, crops_idx, angles

def main(mask_path, source_image_path, target_image_path, output_dir, solver='spsolve', scale=1.0, gradient_mixing_mode='max', gradient_mixing_alpha=1.0):
    # Read input and source images
    # mask = utils.read_image(mask_path, scale=1.0, gray=True)
    src_rgb = utils.read_image(source_image_path, scale=1.0, gray=False)
    target_rgb = utils.read_image(target_image_path, scale=5.0,  gray=False)
    
    # Get the size of the source image
    source_size = src_rgb.shape[:2]

    # Detect edges and smooth the curve
    # edge_curve = detect_edges_and_smooth_curve(target_image_path, scale=10.0)
    spl = detect_edges_and_smooth_curve(target_image_path, scale=5.0)
    # dy_dx = compute_gradient(edge_curve)
    # edge_curve = np.column_stack((edge_curve, dy_dx))
    
    # Crop the input image
    # cropped_images, crop_idx, angles = crop_image(target_rgb, source_size, 50, edge_curve)
    cropped_images, crop_idx, angles = crop_image(target_rgb, source_size, 5, spl)
    # Process each cropped image
    # back_ground = cv2.imread(target_image_path)
    img_maks = cv2.imread(mask_path)
    img_src = cv2.imread(source_image_path)
    for idx, target_image in enumerate(cropped_images):
        mask = rotate_and_resize_image(img_maks, angle=angles[idx], gray=True)
        src_rgb = rotate_and_resize_image(img_src, angle=angles[idx], gray=False)

        # Create a Poisson seamless cloner
        cloner = PoissonSeamlessCloner(mask, src_rgb, target_image, solver, 1.0)
        img = cloner.poisson_blend_rgb("max", 1.0)
        result = target_rgb.copy()
        result[crop_idx[idx][0]:crop_idx[idx][1], crop_idx[idx][2]:crop_idx[idx][3]] = img
        result = (result * 255).astype(np.uint8)
        Image.fromarray(result).save(os.path.join(output_dir, "output_{}.png".format(idx)))
        print("Output image is saved as output_{}.png".format(idx))
        

if __name__ == "__main__":
    mask_path = "generated_mask.png"
    source_image_path = "source.jpg"
    target_image_path = "mountain.jpg"
    output_dir = "output"
    
    main(mask_path, source_image_path, target_image_path, output_dir)