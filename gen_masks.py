import cv2
import numpy as np
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import os

def generate_mask_from_gif(gif_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    gif = Image.open(gif_path)
    frame_count = 0  

    for frame in ImageSequence.Iterator(gif):
        frame_rgb = np.array(frame.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  

        edges = cv2.Canny(gray, threshold1=30, threshold2=200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)  
        cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

        kernel = np.ones((30, 30), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask_filename = os.path.join(output_dir, f"mask_frame_{frame_count:03d}.png")
        cv2.imwrite(mask_filename, mask)

        if frame_count == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.title("Original Frame")
            plt.imshow(frame_rgb)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Edge Detection")
            plt.imshow(edges, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Generated Mask")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        frame_count += 1

    print(f"Processed {frame_count} frames. Masks saved to '{output_dir}'.")

gif_path = "walking_man.gif"  
output_dir = "output_masks"  

generate_mask_from_gif(gif_path, output_dir)
