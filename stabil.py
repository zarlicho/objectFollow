#https://pysource.com/2018/05/14/optical-flow-with-lucas-kanade-method-opencv-3-4-with-python-3-tutorial-31/

import cv2
import numpy as np

cap = cv2.VideoCapture('jump.mp4')

# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Lucas kanade params
lk_params = dict(winSize = (15, 15),
maxLevel = 4,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

point_selected = False
point = ()
old_points = np.array([[]])

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected is True:
        #cv2.circle(frame, point, 5, (0, 0, 255), 2)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points
        
        x, y = new_points.ravel()
        h, w = 310, 313
        crop = frame[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2)]
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        #h1, w1 = crop.shape
        tart = (240, 320)
        end = (x, y)
        color = (0, 255, 0)
  
        # Line thickness of 9 px
        thickness = 9
        
        # Using cv2.line() method
        # Draw a diagonal green line with thickness of 9 px
        h, w = 310, 313
        #print(x1, y1)
        cv2.imshow('crop', crop)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(20)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()