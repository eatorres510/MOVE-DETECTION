import cv2
import numpy as np

# Define motion detection function
def motion_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    difference_frame = cv2.absdiff(gray_frame, blurred_frame)
    thresholded_frame = cv2.threshold(difference_frame, 100, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    return frame

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = motion_detection(frame)
    cv2.imshow("Motion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
