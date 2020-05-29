from gpiozero import Motor
import time
Fright=Motor(9,10)
Fleft=Motor(22,27)
Bright=Motor(3,2)
for i in range(0,4):
    Fright.forward()
    Fleft.forward()
    Bleft.forward()
    Bright.forward()
    time.sleep(5)
    Fright.stop()
    Bleft.stop()
    Fleft.stop()
    Bright.stop()
    Fright.forward() #acually fleft
    Bright.backward()
    Fleft.backward() #actually fright
    Bleft.forward()
    time.sleep(0.9)
    Fright.stop()
    Bleft.stop()
    Fleft.stop()
    Bright.stop()

