import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image, angle, gray=False, plot=False):
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (h, w) = image.shape[:2] 
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2
    final_image = rotated_image[y_start:y_start + h, x_start:x_start + w]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Rotated Image")
        plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    final_image = final_image.astype(np.float64) / 255
    return final_image

# image_path = 'source.jpg'
# image_path = 'generated_mask.png'
# image = cv2.imread(image_path)
# rotated_image = rotate_image(image, angle=45, gray=False, plot=True)