# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
from threading import Thread
from config import *
from picamera2 import Picamera2
import time
""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

class PiHQCamera():
    def __init__(self, resolution=(640, 480), framerate=5):
        self.stopped = False
        self.updateThread = None
        self.resolution = resolution
        self.framerate = framerate
        # self.video_capture = cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=0, capture_width=self.resolution[0], capture_height=self.resolution[1], display_width=self.resolution[0], display_height=self.resolution[1], framerate=self.framerate, flip_method=2), cv2.CAP_GSTREAMER)
        self.video_capture = cv2.VideoCapture(0)
        self.record_flag = False
        self.out = None
        self.frame = None

    def start(self):
        self.stopped = False
        self.updateThread = Thread(target=self.update, args=())
        self.updateThread.start()

    def update(self):
        while not self.stopped:
            # self.frame = self.video_capture.capture_array()
            ret , self.frame = self.video_capture.read()

    def read(self):
        return self.frame

    def stop_thread(self):
        self.stopped = True
        if self.updateThread != None:
            self.updateThread.join()

    def stop(self):
        self.stop_thread()
        # self.video_capture.release()
        del self.video_capture

    def show_camera(self):
        window_title = "CSI Camera"
        try:
            while True:
                ret , self.frame = self.video_capture.read()
                cv2.imshow("Livefeed", self.frame)

                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    self.stop()
                    break
        finally:
            self.stop()
            cv2.destroyAllWindows()
    
    def record_video(self):
        self.record_flag = True

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(DEFAULT_VIDEO_FILENAME, fourcc, self.framerate, self.resolution)

        self.recordThread = Thread(target=self.record_thread, args=())
        self.recordThread.start()

    def record_thread(self):
        while self.record_flag:
            frame = self.video_capture.capture_array()
            self.out.write(frame[:,:,0:3])
    
    def stop_recording_video(self):
        self.record_flag = False
        self.recordThread.join()
        self.out.release()
        

if __name__ == "__main__":
    cam = PiHQCamera(resolution=(1920, 1080),framerate=30)
    # cam.start()
    time.sleep(2)
    cam.show_camera()
    # cam.record_video()
    # time.sleep(2)
    # cam.stop_recording_video()
    # cam.stop()