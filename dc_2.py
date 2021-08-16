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
        frames = pipeline.wait_for_frames()
        
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not aligned_color_frame:
            continue
        
        depth_frame = np.asanyarray(aligned_depth_frame.get_data())
        color_frame = np.asanyarray(aligned_color_frame.get_data())
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        
        images = np.hstack((color_frame, depth_colormap))
        
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        if cv2.waitKey(33) == ord('q'):
            cv2.destroyAllWindows()
            break
        
finally:
    pipeline.stop()