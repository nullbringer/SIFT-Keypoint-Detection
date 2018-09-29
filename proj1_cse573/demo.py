import cv2
import numpy as np


img = cv2.imread("task1.png", 0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite()

# Computing vertical edges
edge_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Eliminate zero values with method 1
pos_edge_x = (edge_x - np.min(edge_x)) / (np.max(edge_x) - np.min(edge_x))
cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Eliminate zero values with method 2
pos_edge_x = np.abs(edge_x) / np.max(np.abs(edge_x))
cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Computing horizontal edges
edge_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
cv2.namedWindow('edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_y_dir', edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Eliminate zero values with method 1
pos_edge_y = (edge_y - np.min(edge_y)) / (np.max(edge_y) - np.min(edge_y))
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Eliminate zero values with method 2
pos_edge_y = np.abs(edge_y) / np.max(np.abs(edge_y))
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

# magnitude of edges (conbining horizontal and vertical edges)
edge_magnitude = np.sqrt(edge_x ** 2 + edge_y ** 2)
edge_magnitude /= np.max(edge_magnitude)
cv2.namedWindow('edge_magnitude', cv2.WINDOW_NORMAL)
cv2.imshow('edge_magnitude', edge_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge_direction = np.arctan(edge_y / (edge_x + 1e-3))
edge_direction = edge_direction * 180. / np.pi
edge_direction /= np.max(edge_direction)
cv2.namedWindow('edge_direction', cv2.WINDOW_NORMAL)
cv2.imshow('edge_direction', edge_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Original image size: {:4d} x {:4d}".format(img.shape[0], img.shape[1]))
print("Resulting image size: {:4d} x {:4d}".format(edge_magnitude.shape[0], edge_magnitude.shape[1]))