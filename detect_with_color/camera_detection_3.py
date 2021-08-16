import cv2
import sys
import time

class CameraDetection:
    
    def __init__(self):
        self.frameWidth = 640
        self.frameHeight = 480
        #self.frameWidth, self.frameHeight = self.frame_size_check()

    def frame_size_check(self):
        capture = cv2.VideoCapture(1)
        if not capture.isOpened():
            print('Camera open failed!')
            sys.exit()
        frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameSize = (frameWidth, frameHeight)
        print("Frame Size : {}".format(frameSize))
        capture.release()
        return frameWidth, frameHeight

    def make_contours(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_unnoise = cv2.bilateralFilter(frame_gray, -1, 10, 5)
        _, frame_binary = cv2.threshold(frame_unnoise, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def find_rectangle(self, frame, contours):
        for points in contours:
            if cv2.contourArea(points) < 1000:
                continue
            approx = cv2.approxPolyDP(points, cv2.arcLength(points, True) * 0.02, True)
            if len(approx) != 4:
                continue
            cv2.polylines(frame, points, True, (0, 0, 255), thickness = 3)
            if cv2.contourArea(points) > self.frameWidth * self.frameHeight * 0.5:
                return True

    def make_return_image(self, frame):
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)
        #cv2.imshow('image_blur', image_blur)
        #cv2.waitKey()
        return image_blur

    def camera_detect(self):
        capture = cv2.VideoCapture(1)
        if not capture.isOpened():
            print('Camera open failed!')
            sys.exit()
        
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Frame read error!")
                sys.exit()
    
            contours = self.make_contours(frame)
            
            if self.find_rectangle(frame, contours) == True:
#                time.sleep(1)
                break
                
            cv2.imshow('Frame', frame)
            if cv2.waitKey(33) == ord('q'):
                break
        
        capture.release()
        cv2.destroyAllWindows()
        
        return self.make_return_image(frame)

detection = CameraDetection()
image = detection.camera_detect()
cv2.imshow('image', image)
cv2.waitKey()
    