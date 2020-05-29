import RPi.GPIO as GPIO
import time
from gpiozero import Motor
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)
pwm=GPIO.PWM(21,100)
pwm.start(5)
angle1=10
duty1=float(angle1)/10 + 2.5
angle2=160
duty2=float(angle2)/10 + 2.5
ck=0
Fright=Motor(9,10)
Fleft=Motor(22,27)
Bright=Motor(3,2)
Bleft=Motor(4,17)
def move_forward():
     Fright.forward()
     Fleft.forward()
     Bleft.forward()
     Bright.forward()
     time.sleep(0.15)
     Fright.stop()
     Bleft.stop()
     Fleft.stop()
     Bright.stop()
def move_backward():
    Fright.backward()
    Fleft.backward()
    Bleft.backward()
    Bright.backward()
    time.sleep(0.5)
    Fright.stop()
    Bleft.stop()
    Fleft.stop()
    Bright.stop()
while ck<1:
    pwm.ChangeDutyCycle(duty2)
    time.sleep(0.8)

    pwm.ChangeDutyCycle(duty1)
    time.sleep(0.8)
    #move_forward()
    pwm.ChangeDutyCycle(duty2)
    time.sleep(0.8)
    #move_backward()
    ck=ck+1


