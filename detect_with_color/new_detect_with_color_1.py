import cv2
import numpy as np                 

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if cv2.contourArea(contour) < areaMin:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
                continue
        cv2.drawContours(imgContour, contour, -1, (255, 0, 255), 7)
        x, y, w, h = cv2.boundingRect(approx)
        center = approx.mean(axis=0)
        center = np.array(center, dtype=np.int16).flatten().tolist()
        print(center)
        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
#        cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(imgContour, "Center: " + str(center), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0), 2)

def empty(a):
    pass

cap = cv2.VideoCapture(0)  

cv2.namedWindow('Parameters')
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 23, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 20, 255, empty)
cv2.createTrackbar("Area" , "Parameters", 1000, 30000, empty)

while True:
    ret, img = cap.read()
    imgContour = img.copy()
    
    imgBlur = cv2.GaussianBlur(img, (0, 0), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    
    getContours(imgDil, imgContour)
    
    cv2.imshow('frame', imgContour)
    cv2.imshow('imgCanny', imgCanny)
    cv2.imshow("imgDil", imgDil)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    