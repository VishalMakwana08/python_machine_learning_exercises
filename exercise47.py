import cv2

# Load the image from the disk
# 'img' becomes a NumPy array representing the pixel data
img = cv2.imread("input.jpg")

# Verify the file was read correctly
# Returns True if the image exists, False if the path is wrong
print("Image loaded:", img is not None)

# Opens a GUI window to visualize the loaded pixel data
cv2.imshow("Original Image", img)

# Save the pixel data currently in 'img' to a new physical file
# imwrite returns a boolean (True/False) indicating if the write was successful
saved = cv2.imwrite("output_copy.jpg", img)

# Print confirmation of the save operation
print("Image saved (output_copy.jpg):", saved)

# Pause the program execution; window stays open until a key is pressed
cv2.waitKey(0)

# Release all window resources from memory
cv2.destroyAllWindows()

