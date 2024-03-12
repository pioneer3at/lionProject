import Jetson.GPIO as GPIO
import time
GPIO.setmode(GPIO.TEGRA_SOC)
mode = GPIO.getmode()
print(mode)

GPIO.setwarnings(False)

channel = 399
GPIO.setup(channel, GPIO.OUT, initial=GPIO.HIGH)

state = GPIO.LOW
while 1:
    GPIO.output(channel, state)
    if state == GPIO.LOW:
        state = GPIO.HIGH

    time.sleep(2)