import cv2
import numpy as np

# Initialize Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# Open video capture
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('cars_video.mp4')
cap = cv2.VideoCapture('traffic.mp4')

# Initialize previous position
prev_pos = None

# Loop over frames
while True:
    # Read frame from video
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold image
    _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours
    for contour in contours:
        # Compute area of contour
        area = cv2.contourArea(contour)

        # # Ignore small contours
        # if area < 150 or area > 2000:
        #     continue
        if area < 100:
            continue

        # Compute centroid of contour
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        pos = np.array([[np.float32(cx)], [np.float32(cy)]])

        # Initialize Kalman filter if this is the first iteration
        if prev_pos is None:
            kalman.statePost = np.array(
                [[np.float32(cx)], [np.float32(cy)], [0], [0]], np.float32)
            prev_pos = pos
            continue

        # Predict state using Kalman filter
        prediction = kalman.predict()

        # Update measurement vector with centroid of contour
        measurement = pos

        # Correct state using Kalman filter
        kalman.correct(measurement)

        # Get updated state vector
        state = kalman.statePost

        # Draw measured position as green rectangle
        h = int((area**0.5)/2)
        cv2.rectangle(frame, (cx-h, cy-h), (cx+h, cy+h), (0, 255, 0), 2)
        #cv2.rectangle(frame, (cx-10, cy-10), (cx+10, cy+10), (0, 255, 0), 2)

        # # Draw predicted position as red circle
        # cv2.circle(frame, (int(prediction[0]), int(
        #     prediction[1])), 5, (0, 0, 255), -1)

        # # Draw measured position as green circle
        # cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # # Draw line between predicted and measured positions
        # cv2.line(frame, (int(prediction[0]), int(
        #     prediction[1])), (cx, cy), (255, 0, 0), 2)

        # Update previous position
        prev_pos = pos

    # Display frame
    cv2.imshow('frame', frame)

    # Exit if 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy window
cap.release()
cv2.destroyAllWindows()
