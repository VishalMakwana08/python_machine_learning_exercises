import cv2

# Load two pre-trained Haar Cascade classifiers
# One for detecting the face structure and another for eye patterns
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the image and convert it to grayscale
# Detection algorithms perform better on single-channel grayscale images
img = cv2.imread("face.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
# 1.1 = scaleFactor (image size reduction), 5 = minNeighbors (quality threshold)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
print("Faces:", len(faces))

# Loop through every face detected in the image
for (x, y, w, h) in faces:
    # Draw a Green (0, 255, 0) rectangle around the detected face on the original image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ROI = Region of Interest. We "crop" the face area specifically.
    # roi_gray is for detection, roi_color is for drawing the eye rectangles
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Search for eyes ONLY within the 'roi_gray' (the cropped face box)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    print("Eyes in this face:", len(eyes))

    # Loop through every eye found within that specific face ROI
    for (ex, ey, ew, eh) in eyes:
        # Draw a Blue (255, 0, 0) rectangle around the eyes
        # Since we use 'roi_color', the (ex, ey) coordinates are relative to the face box
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

# Display the final output with all bounding boxes
cv2.imshow("Face & Eye Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

