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
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.interpolate import interp1d

# # Load the image
# image_path = 'mountain.jpg'
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # print(gray.shape)

# # Step 1: Preprocessing and Canny edge detection
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edges = cv2.Canny(gray, 50, 150)

# # Step 2: Extract the highest edge (topmost white pixel per column)
# height, width = edges.shape
# top_edge_points = []

# for col in range(width):
#     for row in range(height):
#         if edges[row, col] > 0:  # Find the first white pixel in this column
#             top_edge_points.append((col, row))
#             break

# # Convert the points to separate x and y arrays
# x = np.array([pt[0] for pt in top_edge_points])
# y = np.array([pt[1] for pt in top_edge_points])

# # Step 3: Fit a smooth curve using spline interpolation
# x_smooth = np.linspace(x.min(), x.max(), 500)  # Generate 500 smooth points along x-axis
# spl = make_interp_spline(x, y, k=3)  # Spline interpolation of degree 3
# y_smooth = spl(x_smooth)

def detect_edges_and_smooth_curve(image_path, scale):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # scale up the image
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    # Step 1: Preprocessing and Canny edge detection
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

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

    # Apply Gaussian smoothing to the y-coordinates
    y_smoothed = gaussian_filter1d(y, sigma=10.0)  # Adjust sigma for more or less smoothing

    # Step 3: Fit a smooth curve using spline interpolation
    # spl = make_interp_spline(x, y, k=3)  # Spline interpolation of degree 3
    spl = UnivariateSpline(x[::10], y_smoothed[::10], s=10.0)
    
    # x_smooth = np.linspace(x.min(), x.max(), 10000)  # Generate 1000 smooth points along x-axis
    # y_smooth = spl(x_smooth)

    # # Step 4: Plot the results
    # plt.figure(figsize=(10, 6))

    # # Original Image
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis("off")

    # # Detected and Smoothed Edge
    # plt.subplot(1, 2, 2)
    # plt.title("Highest Edge (Smoothed)")
    # plt.imshow(edges, cmap="gray")
    # plt.plot(x_smooth, y_smooth, color='red', linewidth=2)  # Overlay smooth curve
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()

    # # Draw the slope of the curve on the image
    # dy_dx = spl.derivative()
    # slope_smoothed = gaussian_filter1d(-dy_dx(x_smooth), sigma=250.0)  # Smooth the derivative
    # # Transform the slope to degrees
    # slope = np.arctan(slope_smoothed) * 180 / np.pi

    # plt.figure(figsize=(10, 6))
    # plt.title("Slope of the Curve")
    # plt.plot(x_smooth, slope, color='blue', linewidth=2)
    # plt.xlabel("X")
    # plt.ylabel("Slope (dy/dx)")
    # plt.grid()
    # plt.show()


    # # Step 3: Interpolate the smoothed points
    # interp_func = interp1d(x, y_smoothed, kind='linear', fill_value="extrapolate")
    
    # x_smooth = np.linspace(x.min(), x.max(), 100)  # Generate smooth points along x-axis
    # y_smooth = interp_func(x_smooth)

    # # print the x_smooth and y_smooth
    # print('x_smooth:', x_smooth)
    # print('y_smooth:', y_smooth)

    # # Compute the slope (dy/dx) using the gradient of the interpolated points
    # # dy_dx = np.gradient(y_smooth, x_smooth)
    # # Use convolution to compute the gradient
    # # kernel = np.array([1, -8, 0, 8, -1]) / 12
    # kernel = np.array([1, -8, 12, -15, 18, -24, 30, -36, 42, -48, 54, -60, 60, -54, 48, -42, 36, -30, 24, -18, 15, -12, 8, -1]) / 420
    # dy_dx = convolve1d(y_smooth, kernel, mode='nearest') / (x_smooth[1] - x_smooth[0])
    # slope = np.arctan(dy_dx) * 180 / np.pi  # Transform the slope to degrees

    # print('slope:', slope)

    # # Step 4: Plot the results
    # plt.figure(figsize=(10, 6))

    # # Original Image
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis("off")

    # # Detected and Smoothed Edge
    # plt.subplot(1, 2, 2)
    # plt.title("Highest Edge (Smoothed)")
    # plt.imshow(edges, cmap="gray")
    # plt.plot(x_smooth, y_smooth, color='red', linewidth=2)  # Overlay smooth curve
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()

    # # Plot the slope of the curve
    # plt.figure(figsize=(10, 6))
    # plt.title("Slope of the Curve")
    # plt.plot(x_smooth, slope, color='blue', linewidth=2)
    # plt.xlabel("X")
    # plt.ylabel("Slope (degrees)")
    # plt.grid()
    # plt.show()

    # return np.vstack((x_smooth, y_smooth)).T
    return spl

image_path = 'mountain.jpg'
# edge_curve = detect_edges_and_smooth_curve(image_path, 5)
# detect_edges_and_smooth_curve(image_path, 5)
# print(edge_curve)
