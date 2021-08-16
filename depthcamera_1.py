import cv2
import pyrealsense2
import sys
import numpy as np
import realsense_depth as realdepth

capture = realdepth.DepthCamera()
    
while True:
    ret, depth_frame, color_frame = capture.get_frame()
    
    if not ret:
        print("Frame read error!")
        sys.exit()
    
    color_frame_gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    color_frame_unnoise = cv2.bilateralFilter(color_frame_gray, -1, 10, 5)
    _, color_frame_binary = cv2.threshold(color_frame_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(color_frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    depth_frame = np.int8(depth_frame / 10)
    
    for points in contours:
        if cv2.contourArea(points) < 1000:
            continue
        
        approx = cv2.approxPolyDP(points, cv2.arcLength(points, True) * 0.02, True)
        
        if len(approx) != 4:
            continue
        
        cv2.polylines(color_frame, points, True, (0, 0, 255), thickness = 3)
        cv2.polylines(depth_frame, points, True, (-1, -1, -1), thickness = 3)
        
    cv2.imshow('depth_frame', depth_frame)
    cv2.imshow('color_frame', color_frame)
    
    if cv2.waitKey(33) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()