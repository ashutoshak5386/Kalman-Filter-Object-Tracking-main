import cv2
import numpy as np
import webcolors
import csv

# Define color names and their BGR values
color_names = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0),
    'White': (255, 255, 255),
    'Black': (0, 0, 0)
}

# Initialize Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# Open video capture
cap = cv2.VideoCapture('traffic2.mp4')


# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

frame_no = 0
# Initialize CSV file
with open('output.csv', mode='w', newline='') as car_file:
    writer = csv.writer(car_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Frame', 'X', 'Y', 'Color'])
    # Loop over frames
    while True:
        # Read frame from video
        ret, frame = cap.read()
        # Exit if end of video
        if not ret:
            break

        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        # Apply morphology
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over contours
        for contour in contours:
            # Compute area of contour
            area = cv2.contourArea(contour)

            # Ignore small contours
            if area < 100:
                continue

            # Compute centroid of contour
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            pos = np.array([[np.float32(cx)], [np.float32(cy)]])

            # Initialize Kalman filter if this is the first iteration
            if kalman.statePost is None:
                kalman.statePost = np.array(
                    [[np.float32(cx)], [np.float32(cy)], [0], [0]], np.float32)
                continue

            # Predict state using Kalman filter
            prediction = kalman.predict()

            # Update measurement vector with centroid of contour
            measurement = pos

            # Correct state using Kalman filter
            kalman.correct(measurement)

            # Get updated state vector
            state = kalman.statePost

            # Get color of object by computing average color of pixels within contour
            mask = np.zeros_like(fgmask)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_color = cv2.mean(frame, mask=mask)[:3]

            # Compute color difference between measured color and predefined color values
            color_diffs = {name: sum(
                abs(np.array(color_names[name]) - mean_color)) for name in color_names}

            # Find the closest color name based on color difference
            color_name = min(color_diffs, key=color_diffs.get)

            # Draw measured position as green rectangle
            cv2.rectangle(frame, (cx-20, cy-20), (cx+20, cy+20), color_names[color_name], 2)
            cv2.putText(frame, color_name, (cx+20, cy-20),cv2.FONT_HERSHEY_PLAIN, 1, color_names[color_name], 1)
            
            # Write in database
            writer.writerow([frame_no, cx, cy, color_name])

        # Display frame
        cv2.imshow('frame', frame)

        # Exit if 'q' key pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_no +=1
