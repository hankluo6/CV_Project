# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image = cv2.imread('mountain.jpg')

# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # Use Canny edge detection
# edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# # Find contours
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw contours on a blank canvas
# contour_image = np.zeros_like(gray)
# cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=1)

# # Display results
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 3, 1), plt.title('Original Image'), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.subplot(1, 3, 2), plt.title('Edges'), plt.imshow(edges, cmap='gray')
# plt.subplot(1, 3, 3), plt.title('Mountain Edge'), plt.imshow(contour_image, cmap='gray')
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Load the image
image_path = 'mountain.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Preprocessing and Canny edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Step 2: Extract the highest edge (topmost white pixel per column)
height, width = edges.shape
top_edge_points = []

for col in range(width):
    for row in range(height):
        if edges[row, col] > 0:  # Find the first white pixel in this column
            top_edge_points.append((col, row))
            break

# Convert the points to separate x and y arrays
x = np.array([pt[0] for pt in top_edge_points])
y = np.array([pt[1] for pt in top_edge_points])

# Step 3: Fit a smooth curve using spline interpolation
x_smooth = np.linspace(x.min(), x.max(), 500)  # Generate 500 smooth points along x-axis
spl = make_interp_spline(x, y, k=3)  # Spline interpolation of degree 3
y_smooth = spl(x_smooth)

# Step 4: Plot the results
plt.figure(figsize=(10, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Detected and Smoothed Edge
plt.subplot(1, 2, 2)
plt.title("Highest Edge (Smoothed)")
plt.imshow(edges, cmap="gray")
plt.plot(x_smooth, y_smooth, color='red', linewidth=2)  # Overlay smooth curve
plt.axis("off")

plt.tight_layout()
plt.show()
