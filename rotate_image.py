import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Load the image
# # image_path = 'source.jpg'
# image_path = 'generated_mask.png'
# image = cv2.imread(image_path)
# (h, w) = image.shape[:2]  # Get image dimensions

# # Define rotation parameters
# angle = 45  # Rotation angle in degrees (positive = counterclockwise)
# center = (w // 2, h // 2)  # Rotation center (image center)

# # Step 1: Get the rotation matrix
# rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

# # Step 2: Adjust the matrix to keep size
# cos_theta = np.abs(rotation_matrix[0, 0])
# sin_theta = np.abs(rotation_matrix[0, 1])

# # Compute the new bounding dimensions of the image
# new_w = int((h * sin_theta) + (w * cos_theta))
# new_h = int((h * cos_theta) + (w * sin_theta))

# # Adjust the rotation matrix to keep the image centered
# rotation_matrix[0, 2] += (new_w / 2) - center[0]
# rotation_matrix[1, 2] += (new_h / 2) - center[1]

# # Step 3: Perform the rotation
# rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

# # Step 4: Resize back to the original size
# final_image = cv2.resize(rotated_image, (w, h))

# # cv2.imwrite('rotated_image.jpg', final_image)
# cv2.imwrite('rotated_image_mask.jpg', final_image)

# # Display the result
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.title("Rotated Image (Preserved Size)")
# plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# plt.tight_layout()
# plt.show()

def rotate_and_resize_image(image, angle=45, gray=False):
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape[:2]  # Get image dimensions

    # Define rotation parameters
    center = (w // 2, h // 2)  # Rotation center (image center)

    # Step 1: Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Step 2: Adjust the matrix to keep size
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])

    # Compute the new bounding dimensions of the image
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))

    # Adjust the rotation matrix to keep the image centered
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Step 3: Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    # Step 4: Resize back to the original size
    # final_image = cv2.resize(rotated_image, (w, h))
    # Step 4: Crop the image to the original size
    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2
    final_image = rotated_image[y_start:y_start + h, x_start:x_start + w]

    # Normalize the image to match the format of read_image
    final_image = final_image.astype(np.float64) / 255

    return final_image