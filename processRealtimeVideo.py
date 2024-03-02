import cv2
from threading import Thread
from picamera2 import Picamera2
import sys
sys.path.insert(0,"/home/acd/templateMatching")
from editor.videoProcessing import VideoProcessing
import os
import time
from pprint import *
import signal

class PiVideoStream():
    def __init__(self, resolution=(640, 480), framerate=5):
        self.picam2 = Picamera2()
        # self.picam2.configure(self.picam2.create_preview_configuration())
        # self.picam2.start()
        # Run for a second to get a reasonable "middle" exposure level.
        # metadata = self.picam2.capture_metadata()
        # exposure_normal = metadata["ExposureTime"]
        # gain = metadata["AnalogueGain"] * metadata["DigitalGain"]
        # self.picam2.stop()
        # controls = {"ExposureTime": exposure_normal, "AnalogueGain": gain}
        capture_config = self.picam2.create_preview_configuration(main={"size": resolution, "format": "RGB888"})
        self.picam2.configure(capture_config)
        self.picam2.set_controls({"FrameRate": framerate, "ExposureTime": 2000})
        # pprint(self.picam2.sensor_modes)
        self.picam2.start()
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        # Thread(target=self.update, args=()).start()
        x = Thread(target=self.update, args=())
        x.start()

    def update(self):
        while not self.stopped:
            self.frame = self.picam2.capture_array()

    def read(self):
        return self.frame

    def stop(self):
		# indicate that the thread should be stopped
        self.stopped = True
        self.picam2.stop()

class ProcessRealtimeVideo():
    def __init__(self, resolution, framerate, show=False):
        self.vs = PiVideoStream(resolution=resolution,framerate=framerate)
        self.vs.start()
        time.sleep(2)
        self.vp = VideoProcessing()
        self.show = show

    def execute(self):
        count       = 0
        MAX_PERCENT = 0

        TOTAL_QUAN_FOR_JUDGEMENT = 3
        DETECTION_LIST = [False] * TOTAL_QUAN_FOR_JUDGEMENT
        ACCEPTANCE_VALUE = TOTAL_QUAN_FOR_JUDGEMENT - 1

        EVER_DETECTED = False
        EVER_DETECTED_LIST = []
        EVER_DETECTED_MAX_ACCEPTANCE = 0

        while True:
            image = self.vs.read()
            if 0:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # cv2.imwrite(os.path.join("/home/acd/templateMatching/gs/frames", "frame_{}.jpg".format(count)), image)

            start_time = time.time()
            image, ret, percent = self.vp.process_simple(image, self.show)
            if MAX_PERCENT < percent: 
                MAX_PERCENT = percent
            time_consumed   = time.time() - start_time  
            fps             = 1/time_consumed

            DETECTION_LIST.insert(len(DETECTION_LIST) - 1, DETECTION_LIST.pop(0))
            if ret == True:
                DETECTION_LIST[-1] = True
            else:
                DETECTION_LIST[-1] = False

            print("Frame: {} | FPS: {} | Last {}: {}".format(count, round(fps, 2), TOTAL_QUAN_FOR_JUDGEMENT, DETECTION_LIST.count(True)))
            if DETECTION_LIST.count(True) > EVER_DETECTED_MAX_ACCEPTANCE: EVER_DETECTED_MAX_ACCEPTANCE = DETECTION_LIST.count(True)
            if DETECTION_LIST.count(True) >= ACCEPTANCE_VALUE:
                color = (0, 255, 0)
                EVER_DETECTED = True
                EVER_DETECTED_LIST.append(count)
                print('\x1b[6;30;42m' + 'YES!' + '\x1b[0m')

            else:
                print('\x1b[6;33;41m' + 'No!' + '\x1b[0m')

                color = (255,0,0)

            if self.show:
                cv2.putText(image, "FRAME: {} | fps: {} | Detection: {}/{}".format(count, round(fps, 2), DETECTION_LIST.count(True), TOTAL_QUAN_FOR_JUDGEMENT), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 2)
                cv2.putText(image, str(ret), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 2)
                cv2.putText(image, str(percent), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 2)
                cv2.putText(image, "EVER_DETECTED: {} @ frame {}".format(EVER_DETECTED, EVER_DETECTED_LIST), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 2)
                cv2.putText(image, "RESOLUTION: {}".format(image.shape), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 2)
                
                cv2.imshow("Camera", image)
                if (cv2.waitKey(25) & 0xFF) == ord('q'):
                    cv2.destroyAllWindows()
                    self.stop()
                    break 

            count += 1

    def stop(self):
        self.vs.stop()

# print("MAX_PERCENT", MAX_PERCENT)
# print("EVER_DETECTED_MAX_ACCEPTANCE", EVER_DETECTED_MAX_ACCEPTANCE)
# print("EVER DETECTED", EVER_DETECTED)
# if DETECTED_PERCENTS != []:
#     print("Average DETECTED_PERCENTS: {} | Quan: {}".format(round(np.mean(DETECTED_PERCENTS),2), len(DETECTED_PERCENTS)))

# vs = PiVideoStream()
# vs.start()
# time.sleep(2)
# count=0
# while True:
#     image = vs.read()
#     cv2.imwrite(os.path.join("/home/acd/templateMatching/gs/frames", "frame_{}.jpg".format(count)), image)
#     # cv2.imshow("Camera", image)
#     # if cv2.waitKey(1) == ord('q'):
#     #     vs.stop()
#     #     break
#     count += 1

if __name__ == '__main__':
    try:
        processor = ProcessRealtimeVideo(resolution=(800, 600), framerate=5, show=True)
        processor.execute()
    except KeyboardInterrupt:
        processor.stop()