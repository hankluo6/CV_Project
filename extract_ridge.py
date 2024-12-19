import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

def extract_ridge_line(image_path, scale, plot=False):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    height, width = edges.shape
    top_edge_points = []
    for col in range(width):
        for row in range(height):
            if edges[row, col] > 0:
                top_edge_points.append((col, row))
                break

    x = np.array([pt[0] for pt in top_edge_points])
    y = np.array([pt[1] for pt in top_edge_points])

    y_smoothed = gaussian_filter1d(y, sigma=10.0)

    # spl = make_interp_spline(x, y, k=3)
    spl = UnivariateSpline(x[::10], y_smoothed[::10], s=10.0)
    
    if plot:
        x_data = np.linspace(x.min(), x.max(), 1000)
        y_data = spl(x_data)

        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Highest Edge")
        plt.imshow(edges, cmap="gray")
        plt.plot(x_data, y_data, color='red', linewidth=2)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return spl

# image_path = 'mountain.jpg'
# spl = extract_ridge_line(image_path, 5, plot=True)