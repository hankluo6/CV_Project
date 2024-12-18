import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_mask(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Step 2: Edge detection using Canny
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Step 3: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        print(f"Contour {i} shape: {contour.shape}")

    # Step 4: Create an empty mask and fill the contours
    mask = np.zeros_like(gray)  # Black mask same size as the grayscale image
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

    # # Step 5: Apply morphological closing to smooth the mask
    # kernel = np.ones((5, 5), np.uint8)  # Define the kernel size
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # # Display the results
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title("Edge Detection")
    # plt.imshow(edges, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title("Generated Mask")
    # plt.imshow(mask, cmap='gray')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

     # Step 5: Morphological closing to close small holes
    kernel = np.ones((7, 7), np.uint8)  # Larger kernel to close gaps
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 6: Flood fill to fill inner holes
    mask_filled = mask_closed.copy()
    h, w = mask_closed.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_filled, flood_mask, (0, 0), 255)  # Fill from the top-left corner

    # Invert the flood-filled area and combine with the original mask
    mask_inverted = cv2.bitwise_not(mask_filled)
    final_mask = cv2.bitwise_or(mask_closed, mask_inverted)

    # Display results
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
    plt.title("Refined Mask")
    plt.imshow(mask_closed, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Final Mask (Filled)")
    plt.imshow(final_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Optionally, save the mask
    cv2.imwrite('generated_mask.png', mask)

# Path to your input image
image_path = 'source.jpg'  # Replace with your image path
generate_mask(image_path)
