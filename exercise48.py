import cv2

# Load the pre-trained Haar Cascade model for frontal face detection
# cv2.data.haarcascades provides the path to the built-in XML model files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read the image file
img = cv2.imread("face.jpg")

# Convert to grayscale because face detection algorithms typically 
# process intensity (brightness) rather than color to save computation time
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
# scaleFactor: How much the image size is reduced at each image scale
# minNeighbors: How many neighbors each candidate rectangle should have to retain it
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Output the number of detected face objects found in the 'faces' list
print("Total faces detected:", len(faces))

# Iterate through the list of detected faces
# Each face is represented as (x, y, w, h) - coordinates and size
for (x, y, w, h) in faces:
    # Draw a green rectangle around the face
    # (0, 255, 0) is Green in BGR, and '2' is the line thickness
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the final result with bounding boxes
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

