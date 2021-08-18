import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        #start
        frames = pipeline.wait_for_frames()
        
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not aligned_color_frame:
            continue
        
        depth_frame = np.asanyarray(aligned_depth_frame.get_data())
        color_frame = np.asanyarray(aligned_color_frame.get_data())
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        #end
        color_frame_gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        color_frame_unnoise = cv2.bilateralFilter(color_frame_gray, -1, 10, 5)
        _, color_frame_binary = cv2.threshold(color_frame_unnoise, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(color_frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        images = np.hstack((color_frame_gray, depth_colormap))
        for points in contours:
            if cv2.contourArea(points) < 1000:
                continue
            approx = cv2.approxPolyDP(points, cv2.arcLength(points, True) * 0.02, True)
            if len(approx) != 4:
                continue
            cv2.polylines(color_frame, points, True, (0, 0, 255), thickness = 3)
            cv2.polylines(depth_colormap, points, True, (0, 0, 255), thickness = 3)
            center = approx.reshape(4, 2).astype(np.uint16).mean(axis=0, dtype=np.uint16)

        cv2.imshow('color', color_frame)
        cv2.imshow('depth', depth_colormap)
        if cv2.waitKey(33) == ord('q'):
            cv2.destroyAllWindows()
            break
        
finally:
    pipeline.stop()