import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_mask(image_path, plot=False):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = [max(contours, key=cv2.contourArea)]

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

    # kernel = np.ones((5, 5), np.uint8)
    kernel = np.ones((7, 7), np.uint8)  # Larger kernel to close gaps
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Flood fill to fill inner holes
    mask_filled = mask_closed.copy()
    h, w = mask_closed.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_filled, flood_mask, (0, 0), 255)

    # Invert the flood-filled area and combine with the original mask
    mask_inverted = cv2.bitwise_not(mask_filled)
    mask_final = cv2.bitwise_or(mask_closed, mask_inverted)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Mask")
    plt.imshow(mask_closed, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Filled Mask")
    plt.imshow(mask_final, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite('mask.jpg', mask_final)

image_path = 'source.jpg'
generate_mask(image_path, plot=True)