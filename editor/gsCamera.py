import cv2
from threading import Thread
from picamera2 import Picamera2
import sys
# sys.path.insert(0,"/home/acd/templateMatching")
# from editor.videoProcessing import VideoProcessing
import os
import time
from pprint import *
import signal

class GsCamera():
    def __init__(self, resolution=(640, 480), framerate=5):
        self.picam2 = Picamera2()
        capture_config = self.picam2.create_preview_configuration(main={"size": resolution, "format": "RGB888"})
        self.picam2.configure(capture_config)
        self.picam2.set_controls({"FrameRate": framerate, "ExposureTime": 10000})
        # pprint(self.picam2.sensor_modes)
        self.picam2.start()
        self.stopped = False
        self.updateThread = None

    def start(self):
        self.stopped = False
        self.updateThread = Thread(target=self.update, args=())
        self.updateThread.start()

    def update(self):
        while not self.stopped:
            self.frame = self.picam2.capture_array()

    def read(self):
        return self.frame

    def stop_thread(self):
        self.stopped = True
        if self.updateThread != None:
            self.updateThread.join()

    def stop(self):
        self.stopped = True
        if self.updateThread != None:
            self.updateThread.join()
        self.picam2.stop()
