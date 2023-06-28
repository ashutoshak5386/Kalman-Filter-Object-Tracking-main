import cv2
import numpy as np

# Set up video capture
cap = cv2.VideoCapture('cars_video.mp4')
#cap = cv2.VideoCapture(0)

# Create Kalman filter object
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)
kf.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If reached video end
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Update Kalman filter with new measurement
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        kf.correct(np.array([[x+w/2], [y+h/2]], dtype=np.float32))

    # Predict next state of the filter
    prediction = kf.predict()
    x, y = prediction[0][0], prediction[1][0]

    # Draw predicted position on the frame
    cv2.circle(frame, (int(x), int(y)), 10, (0,0,255), -1)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    
    
    # Exit if 'q' key pressed
    if cv2.waitKey(1) == ord('q'):
        break
