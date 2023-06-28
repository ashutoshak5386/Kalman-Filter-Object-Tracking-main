import cv2

#cap = cv2.VideoCapture('cars_video.mp4')
cap = cv2.VideoCapture('traffic.mp4')
#cap = cv2.VideoCapture(0)
# Create a background subtractor object
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    # If reached video end
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.erode(fgMask, kernel)
    fgMask = cv2.dilate(fgMask, kernel)

    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # area = cv2.contourArea(contours)
    # if area<100:
    #     continue
    
    # Draw bounding boxes around the contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Moving Object Detection', frame)
    
    
    
    
    # Press esc key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
