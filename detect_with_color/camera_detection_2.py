import cv2
import sys

def frame_size_check(capture):
    frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameSize = (frameWidth, frameHeight)
    print("Frame Size : {}".format(frameSize))
    return frameWidth, frameHeight

def make_contours(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame_binary = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

def find_rectangle(frame, contours, frameWidth, frameHeight):
    for points in contours:
        if cv2.contourArea(points) < 1000:
            continue
        
        approx = cv2.approxPolyDP(points, cv2.arcLength(points, True) * 0.02, True)
        
        if len(approx) != 4:
            continue
        
        cv2.polylines(frame, points, True, (0, 0, 255), thickness = 1)
        
        if cv2.contourArea(points) > frameWidth * frameHeight * 0.5:
            return True

def make_return_image(frame):
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)
    #cv2.imshow('image_blur', image_blur)
    #cv2.waitKey()
    
    return image_blur

def camera_detection():
    frameWidth = 640
    frameHeight = 480
    
    capture = cv2.VideoCapture(1)
    if not capture.isOpened():
        print('Camera open failed!')
        sys.exit()
    
    #frameWidth, frameHeight = frame_size_check(capture)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Frame read error!")
            sys.exit()

        contours = make_contours(frame)
        
        if find_rectangle(frame, contours, frameWidth, frameHeight) == True:
            break
            
        cv2.imshow('Frame', frame)
        if cv2.waitKey(33) == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
    
    return make_return_image(frame)

image = camera_detection()
#cv2.imshow('image', image)
#cv2.waitKey()
    