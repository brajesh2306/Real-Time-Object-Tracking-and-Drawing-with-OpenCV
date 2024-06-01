import cv2
import numpy as np
cap = cv2.VideoCapture(0)

lower_color = np.array([100, 150, 0])
upper_color = np.array([140, 255, 255])

drawing = np.zeros((480, 640, 3), dtype=np.uint8)

prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            cv2.circle(frame, center, 20, (0, 255, 0), 2)

            if prev_center is not None:
                cv2.line(drawing, prev_center, center, (0, 255, 0), 5)
            prev_center = center
        else:
            prev_center = None
    else:
        prev_center = None
    frame = cv2.add(frame, drawing)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(30) & 0xFF == ord('b'):
        break
cap.release()
cv2.destroyAllWindows()