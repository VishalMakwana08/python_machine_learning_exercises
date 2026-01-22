import cv2

# Load two pre-trained classifiers: one for faces and one for eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the image and convert to grayscale for faster processing
img = cv2.imread("face.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect all faces in the image
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
print("Faces:", len(faces))

# Loop through every face found
for (x, y, w, h) in faces:
    # Draw a Green (0, 255, 0) rectangle around the detected face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ROI = Region of Interest. We "crop" the face area from the image.
    # We do this in both grayscale (for detection) and color (for drawing)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Search for eyes ONLY within the 'roi_gray' (the face area)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    print("Eyes in this face:", len(eyes))

    # Loop through every eye found within that specific face
    for (ex, ey, ew, eh) in eyes:
        # Draw a Blue (255, 0, 0) rectangle around the eyes
        # Note: We draw on 'roi_color' so the coordinates are relative to the face
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

# Display the final output
cv2.imshow("Face & Eye Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

