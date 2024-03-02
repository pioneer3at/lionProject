# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
from threading import Thread
from config import *

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

# class HQCameraOrin():
#     def __init__(self, resolution=(1920, 1080), framerate=30):
#     	self.gstreamer_pipeline(sensor_id=1, capture_width=resolution[0], capture_height=resolution[1], display_width=960, display_height=540, framerate=30, flip_method=0)
    	
# 	def gstreamer_pipeline(
# 		self,
# 	    sensor_id=0,
# 	    capture_width=1920,
# 	    capture_height=1080,
# 	    display_width=960,
# 	    display_height=540,
# 	    framerate=30,
# 	    flip_method=0,
# 	):
# 	    return (
# 		"nvarguscamerasrc sensor-id=%d ! "
# 		"video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
# 		"nvvidconv flip-method=%d ! "
# 		"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
# 		"videoconvert ! "
# 		"video/x-raw, format=(string)BGR ! appsink"
# 		% (
# 		    sensor_id,
# 		    capture_width,
# 		    capture_height,
# 		    framerate,
# 		    flip_method,
# 		    display_width,
# 		    display_height,
# 		)
# 	    )


# def show_camera():
#     window_title = "CSI Camera"

#     # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
#     print(gstreamer_pipeline(flip_method=0))
#     video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
#     if video_capture.isOpened():
#         try:
#             window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
#             while True:
#                 ret_val, frame = video_capture.read()
#                 # Check to see if the user closed the window
#                 # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
#                 # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
#                 if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
#                     cv2.imshow(window_title, frame)
#                 else:
#                     break 
#                 keyCode = cv2.waitKey(10) & 0xFF
#                 # Stop the program on the ESC key or 'q'
#                 if keyCode == 27 or keyCode == ord('q'):
#                     break
#         finally:
#             video_capture.release()
#             cv2.destroyAllWindows()
#     else:
#         print("Error: Unable to open camera")


# if __name__ == "__main__":
#     show_camera()
    
    
class OrinHQCamera():
    def __init__(self, resolution=(640, 480), framerate=5):
        self.stopped = False
        self.updateThread = None
        self.resolution = resolution
        self.framerate = framerate
        self.video_capture = cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=0, capture_width=self.resolution[0], capture_height=self.resolution[1], display_width=self.resolution[0], display_height=self.resolution[1], framerate=self.framerate, flip_method=2), cv2.CAP_GSTREAMER)
        self.record_flag = False
        self.out = None

    def gstreamer_pipeline(
    	self,
        sensor_id=0,
        capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=0,
    ):
        return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1, auto-exposure=1, exposure-time=.00001! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
        )

    def start(self):
        self.stopped = False
        self.updateThread = Thread(target=self.update, args=())
        self.updateThread.start()

    def update(self):
        while not self.stopped:
            if self.video_capture.isOpened():
                ret_val, self.frame = self.video_capture.read()

    def read(self):
        return self.frame

    def stop_thread(self):
        self.stopped = True
        if self.updateThread != None:
            self.updateThread.join()

    def stop(self):
        self.stop_thread()
        self.video_capture.release()

    def show_camera(self):
        window_title = "CSI Camera"
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, self.read())
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    self.stop()
                    break
        finally:
            self.video_capture.release()
            cv2.destroyAllWindows()
    
    def record_video(self):
        self.record_flag = True

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(DEFAULT_VIDEO_FILENAME, fourcc, self.framerate, self.resolution)

        self.recordThread = Thread(target=self.record_thread, args=())
        self.recordThread.start()

    def record_thread(self):
        while self.video_capture.isOpened() and self.record_flag:
            ret_val, frame = self.video_capture.read()
            if ret_val:
                self.out.write(frame)
            else:
                break
    
    def stop_recording_video(self):
        self.record_flag = False
        self.recordThread.join()
        self.out.release()
        

if __name__ == "__main__":
    cam = OrinHQCamera(resolution=(1920, 1080),framerate=30)
    cam.start()
    cam.show_camera()