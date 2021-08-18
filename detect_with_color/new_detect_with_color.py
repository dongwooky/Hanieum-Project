import cv2
import numpy as np

cap = cv2.VideoCapture(0)                    

def empty(a):
    pass

cv2.nameddWindow('Parameters')
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)

while True:
    ret, img = cap.read()
    
    imgBlur = cv2.GaussianBlur(img, (0, 0), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', img)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    