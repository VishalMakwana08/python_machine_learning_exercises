import cv2
import os

# 1. Use an absolute path if the image is in a different folder
# Note: Use double backslashes \\ or a forward slash / in Python paths
file_path = "input.jpg" 

# Check if file actually exists before trying to read it
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found in the directory.")
    print("Current Working Directory:", os.getcwd())
else:
    img = cv2.imread(file_path)

    # 2. Check if the image was successfully loaded
    if img is None:
        print("Error: Could not decode the image. It might be corrupted.")
    else:
        print("Image loaded successfully!")

        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print("Original shape (H,W,Channels):", img.shape)
        print("Gray shape (H,W):", gray.shape)

        # Create windows to display
        cv2.imshow("Original", img)
        cv2.imshow("Gray", gray)

        cv2.waitKey(0)
        cv2.destroyAllWindows()