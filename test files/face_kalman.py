import cv2
import numpy as np

# Initialize Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# Load object detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open video capture
cap = cv2.VideoCapture(0)

# Loop over frames
while True:
    # Read frame from video
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Predict state using Kalman filter
        prediction = kalman.predict()

        # Update measurement vector with detected face position
        measurement = np.array([[np.float32(x + w/2)], [np.float32(y + h/2)]])

        # Correct state using Kalman filter
        kalman.correct(measurement)

        # Get updated state vector
        state = kalman.statePost

        # Draw face bounding box
        cv2.rectangle(frame, (int(x), int(y)),
                      (int(x+w), int(y+h)), (0, 255, 0), 2)

        # Draw Kalman filter prediction
        cv2.circle(frame, (int(prediction[0]), int(
            prediction[1])), 5, (0, 0, 255), -1)

    # Display frame
    cv2.imshow('frame', frame)

    # Exit if 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
