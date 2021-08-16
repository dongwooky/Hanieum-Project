import cv2
import sys

frameWidth = 640
frameHeight = 480
frameSize = (frameWidth, frameHeight)

capture = cv2.VideoCapture(1)

if not capture.isOpened():
    print('Camera open failed!')
    sys.exit()

'''  
frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameSize = (frameWidth, frameHeight)
print("Frame Size : {}".format(frameSize))
'''

capture.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

while True:
    ret, frame = capture.read()
    
    if not ret:
        print("Frame read error!")
        sys.exit()
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_unnoise = cv2.bilateralFilter(frame_gray, -1, 10, 5)
    _, frame_binary = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for points in contours:
        if cv2.contourArea(points) < 1000:
            continue
        
        approx = cv2.approxPolyDP(points, cv2.arcLength(points, True) * 0.02, True)
        
        if len(approx) != 4:
            continue
        
        cv2.polylines(frame, points, True, (0, 0, 255), thickness = 1)
        
    cv2.imshow('Frame', frame)
    cv2.imshow('frame_gray', frame_gray)
    cv2.imshow('frame_unnoise', frame_unnoise)
    cv2.imshow('frame_binary', frame_binary)
    
    if cv2.waitKey(33) == ord('q'):
        break
    elif cv2.contourArea(points) > frameWidth * frameHeight * 0.9:
        break
    
image_blur = cv2.GaussianBlur(frame_gray, (3, 3), 0)
cv2.imshow('image_blur', image_blur)
cv2.waitKey()
'''
text = pyt.image_to_string(image_blur, lang = 'Hangul+eng')
print(text)
'''
capture.release()
cv2.destroyAllWindows()
    
    