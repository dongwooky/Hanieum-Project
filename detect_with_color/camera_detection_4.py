import cv2
import sys
import time
import numpy as np

class CameraDetection:
    
    def __init__(self):
        self.frameWidth = 640   #카메라 크기 초기화
        self.frameHeight = 480
        #self.frameWidth, self.frameHeight = self.frame_size_check()    #카메라 사이즈 체크
        self.frame = np.zeros((self.frameHeight, self.frameWidth), np.uint8)    #카메라 프레임 초기화
        self.approx_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32) #꼭지점 배열 초기화
        
        self.srcQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)   #Perspective source 점 4개 초기화
        self.dstQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)   #Perspecrive 점 4개 초기화
        
        self.dstWidth = 0   #펼칠 프레임 크기 초기화
        self.dstHeight = 0
    
    #프레임 크기 확인
    def frame_size_check(self):
        capture = cv2.VideoCapture(1)   #카메라 가져오기
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
            self.approx_points = approx
            cv2.polylines(frame, points, True, (0, 0, 255), thickness = 3)
#            if cv2.contourArea(points) > self.frameWidth * self.frameHeight * 0.5:
#                return True         
            return False
            
    def reorder_points(self, points):
        idx = np.lexsort((points[:, 1], points[:, 0]))
        points = points[idx]
        
        if points[0, 1] > points[1, 1]:
            points[[0, 1]] = points[[1, 0]]
            
        if points[2, 1] < points[3, 1]:
            points[[2, 3]] = points[[3, 2]]
        
        self.dstWidth = round(2 * (points[3, 0] - points[0, 0]))
        self.dstHeight = round(2 * (points[1, 1] - points[0, 1]))
        
        self.dstQuad = np.array([[0, 0], [0, self.dstHeight], 
                                 [self.dstWidth, self.dstHeight], [self.dstWidth, 0]], np.float32)
        
        return points
    
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
            ret, self.frame = capture.read()
            if not ret:
                print("Frame read error!")
                sys.exit()
    
            contours = self.make_contours(self.frame)
            
            is_rectangle = self.find_rectangle(self.frame, contours)
            
            if is_rectangle == True:
                break
                
            cv2.imshow('Frame', self.frame)
            if cv2.waitKey(33) == ord('q'):
                break
            
        self.srcQuad = self.reorder_points(self.approx_points.reshape(4, 2).astype(np.float32))      
        perspective = cv2.getPerspectiveTransform(self.srcQuad, self.dstQuad)
        dst_frame = cv2.warpPerspective(self.frame, perspective, (self.dstWidth, self.dstHeight))
        
        capture.release()
        cv2.destroyAllWindows()
        return dst_frame  
      
#        return self.make_return_image(dst_frame)

detection = CameraDetection()
image = detection.camera_detect()
cv2.imshow('image', image)
cv2.waitKey()
cv2.destoryAllWindows()
    