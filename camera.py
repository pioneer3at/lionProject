#!/usr/bin/python3
import time

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
import cv2
from libcamera import Transform
from pprint import *
### Preview ###
# picam2 = Picamera2()
# config = picam2.create_preview_configuration()
# picam2.configure(config)
# picam2.start()
# time.sleep(2)
# picam2.stop_preview()
# picam2.start_preview(True)
# time.sleep(200)
###########

picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (1456, 1088)}, lores={"size": (640,480)}, encode="lores")
picam2.configure(video_config)
picam2.set_controls({"FrameRate": 15, "ExposureTime": 5000})

encoder = H264Encoder(10000000)
pprint(picam2.sensor_modes)
# ### Picture
picam2.start()
time.sleep(2)
picam2.capture_file("test.jpg")

### Video
# picam2.start_recording(encoder, 'video_correct_3.h264')
# picam2.start_recording(encoder, 'video.h264')
# time.sleep(5)
# picam2.stop_recording()

# picam2.encoder = encoder
# picam2.start_preview()
# picam2.start()
# picam2.start_encoder(encoder)

# while True:
#     cur = picam2.capture_buffer()
#     cv2.imshow('frame', cur)
#     if cv2.waitKey(1) == ord('q'):
#         break
