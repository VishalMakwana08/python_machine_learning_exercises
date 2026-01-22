import cv2
import numpy as np # Import NumPy for numerical array operations

# Load the image into a NumPy array
img = cv2.imread("input.jpg")
print("Loaded:", img is not None)

# Using numpy to get basic statistics from image pixels
# .shape returns (Rows/Height, Columns/Width, Color Channels)
print("Height, Width, Channels:", img.shape)

# .min() and .max() find the lowest and highest brightness values (0-255)
print("Min pixel value:", img.min())
print("Max pixel value:", img.max())

# .mean() calculates the average brightness across all pixels and channels
print("Mean pixel value:", round(img.mean(), 2))

# Create a blank image using numpy (black image)
# np.zeros creates an array filled with 0s (black) 
# uint8 defines the data type as 8-bit unsigned integers (values 0-255)
blank = np.zeros((300, 300, 3), dtype=np.uint8)

# Draw text on the blank image
# Arguments: (image, text, coordinates, font, scale, color_in_BGR, thickness)
cv2.putText(blank, "OpenCV", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Display the images in separate windows
cv2.imshow("Original", img)
cv2.imshow("Blank with Text", blank)

# Wait for a key press before closing windows
cv2.waitKey(0)

# Clean up memory by closing all GUI windows
cv2.destroyAllWindows()

