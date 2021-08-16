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
    
#    depth_frame = np.int8(depth_frame / 2)
    depth_frame = np.where(depth_frame >= 500, 0, depth_frame).astype('uint8')
    depth_frame = np.where(depth_frame >= 465, 85, depth_frame).astype('uint8')
    depth_frame = np.where(depth_frame >= 435, 170, depth_frame).astype('uint8')
    depth_frame = np.where(depth_frame >= 465, 255, depth_frame).astype('uint8')
    
    cv2.imwrite('./data/depth_frame.png', depth_frame)
    depth_frame_png = cv2.imread('./data/depth_frame.png')
    depth_frame_gray = cv2.cvtColor(depth_frame_png, cv2.COLOR_BGR2GRAY)
    _, depth_frame_binary = cv2.threshold(depth_frame_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(depth_frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for points in contours:
        if cv2.contourArea(points) < 500:
            continue
        
        approx = cv2.approxPolyDP(points, cv2.arcLength(points, True) * 0.02, True)
        
        if len(approx) != 4:
            continue
        
        cv2.polylines(depth_frame_gray, points, True, 255, thickness = 3)

    cv2.imshow('depth_frame', depth_frame)
#    cv2.imshow('depth_frame_gray', depth_frame_gray)
#    cv2.imshow('depth_frame_median', depth_frame_median)
#    cv2.imshow('depth_frame_binary', depth_frame_binary)
#    cv2.imshow('depth_frame_binary_median', depth_frame_binary_median)
#    cv2.imshow('color_frame', color_frame)
    
    if cv2.waitKey(33) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()