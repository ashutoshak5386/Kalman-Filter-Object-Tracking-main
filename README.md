# Kalman Filter Object Tracking  

The Kalman Filter Object Tracking project is a computer vision project that utilizes the Kalman Filter algorithm to track objects in a recorded or live video. This project can be used in surveillance, object detection, and tracking in real-time systems. 

## Background

The Kalman Filter is a mathematical algorithm that is widely used in control systems and signal processing to filter out noise and uncertainty from measurements. The Kalman Filter is especially useful in situations where measurements are incomplete or inaccurate. The Kalman Filter algorithm was first developed by Rudolf Emil Kalman in 1960.

## Object Tracking

Object tracking is a process of locating objects in consecutive video frames. It involves a variety of computer vision techniques such as background subtraction, contour detection, and motion estimation. The goal of object tracking is to follow a moving object through a video sequence and to determine its location in each frame.

## Color Detection

Color detection is a process of recognizing colors in images or videos. This is useful in a variety of applications, such as object detection, image processing, and computer vision. In this project, the closest color of the object is detected from the given set of colors - black, white, red, green, blue.

## Implementation

The Kalman Filter Object Tracking project has been implemented using Python and OpenCV libraries. The project consists of two main modules - main.py and app.py. 

* **main.py** - This module performs the object tracking and color detection. It utilizes the OpenCV library to detect and track moving objects in a video. It also determines the closest color of the object and records all the instances of objects being tracked along with their position and color in a CSV file named "output.csv". 

* **app.py** - This module contains a Flask web application that serves a single page named "index.html". The Flask application provides an endpoint that executes the main.py file and another endpoint that reads the "output.csv" file and sends the data to the webpage for real-time display.

## Usage

To use the Kalman Filter Object Tracking project, follow these steps:

1. Install Python and OpenCV libraries.
2. Clone the project repository.
3. Run the "app.py" file to start the Flask application.
4. Open a web browser and navigate to "http://localhost:5000".
5. Click the "Start" button on the webpage to start object tracking.
6. Objects being tracked are displayed in a table format on the webpage, which updates in real-time with every new data point.

## Conclusion

The Kalman Filter Object Tracking project provides a useful tool for object tracking and color detection. This project can be used in a variety of real-world applications, such as video surveillance, motion tracking, and object detection. The project's implementation in Python and OpenCV libraries makes it easy to integrate with other applications and systems.
