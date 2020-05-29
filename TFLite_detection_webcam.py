######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
from threading import Thread
import importlib.util
import math
import speech_recognition as sr
from gpiozero import Motor
import time
import wiringpi
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)
Fright=Motor(9,10)
Fleft=Motor(22,27)
Bright=Motor(3,2)
Bleft=Motor(4,17)
mic_name="USB PnP Sound Device: Audio (hw:0,0)"
sample_rate=44100
chunk_size=512
label_out=[]
mid_x_out=[]
mid_y_out=[]
label1=[]
mid_x1=[]
mid_y1=[]
h=0
w=0

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
def object_detection():
    label_out=[]
    mid_x_out=[]
    mid_y_out=[]
    class VideoStream:
        """Camera object that controls video streaming from the Picamera"""
        def __init__(self,resolution=(640,480),framerate=30):
            # Initialize the PiCamera and the camera image stream
            self.stream = cv2.VideoCapture(0)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3,resolution[0])
            ret = self.stream.set(4,resolution[1])
                
            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
            self.stopped = False

        def start(self):
        # Start the thread that reads frames from the video stream
            Thread(target=self.update,args=()).start()
            return self

        def update(self):
            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

        def read(self):
        # Return the most recent frame
            return self.frame

        def stop(self):
        # Indicate that the camera and thread should be stopped
            self.stopped = True

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    # Import TensorFlow libraries
    # If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tensorflow')
    if pkg is None:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    while True:
        flag=0
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                cv2.circle(frame, (xmin,ymin), 5, (255, 255, 0), cv2.FILLED)
                cv2.circle(frame, (xmax,ymax), 5, (0, 255, 255), cv2.FILLED)
                x_diff=xmax - xmin
                y_diff=ymax - ymin
                mid_x=x_diff/2 + xmin
                mid_x=math.ceil(mid_x)
                mid_y=ymin+y_diff/2
                mid_y=math.ceil(mid_y)
#                 print("xmin=%d" % xmin)
#                 print("ymin=%d" % ymin)
#                 print("xmax=%d" % xmax)
#                 print("ymax=%d" % ymax)
#                 print("mid_x=%d" % mid_x)
#                 print("mid_y=%d" % mid_y)
                cv2.circle(frame, (0,0), 5, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (mid_x,mid_y), 5, (255, 255, 255), cv2.FILLED)
                 
                
                #cv2.circle(frame, (math.ceil(w/2),math.ceil(h/2)), 7, (255,255,255), cv2.FILLED)
#                 print("h=%d" % math.ceil(h/2))
#                 print("w=%d" % math.ceil(w/2))
#                 if mid_x>w/2:
#                     print("right")
#                 else:
#                     print("left")
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                label_out.append(label)
                mid_x_out.append(mid_x)
                mid_y_out.append(mid_y)
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        #cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
#         if cv2.waitKey(1)== ord('q'):
#             break
        #cv2.waitKey(1000)
        #print(label_out)
        #print(mid_x_out)
        #print(mid_y_out)
        (h,w)=frame.shape[:2]
#         for i in label_out:
#             print(i)
#             if "cup" in i:
#                  while(mid_x_out[0]!=w/2):  
#                     if mid_x_out[0]>w/2:
#                         print("object is to the right side of screen")
#                         dist=mid_x_out[0]-w/2
#                         Fright.backward() #acually fleft
#                         Bright.forward()
#                         Fleft.forward() #actually fright
#                         Bleft.backward()
#                         t1=dist*0.05
#                         time.sleep(t1)
#                         Fright.stop()
#                         Bleft.stop()
#                         Fleft.stop()
#                         Bright.stop()
#                     else:
#                             print("obj is to the left side of frame")
#                             dist=mid_x_out[0]-w/2
#                             Fright.forward() #acually fleft
#                             Bright.backward()
#                             Fleft.backward() #actually fright
#                             Bleft.forward()
#                             t1=dist*0.05
#                             time.sleep(t1)
#                             Fright.stop()
#                             Bleft.stop()
#                             Fleft.stop()
#                             Bright.stop()
                        #flag=1
             #move_forward()
             #scan(speech_input)
        cv2.waitKey(100)
        break
        #if(flag==1):
            
            #break
    
    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    return(label_out,mid_x_out,mid_y_out,h/2,w/2)
#label1,mid_x1,mid_y1,h,w=object_detection()
#print(label1)
#print(mid_x1)
#print(mid_y1)
def object_detection_loop():
    class VideoStream:
        """Camera object that controls video streaming from the Picamera"""
        def __init__(self,resolution=(640,480),framerate=30):
            # Initialize the PiCamera and the camera image stream
            self.stream = cv2.VideoCapture(0)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3,resolution[0])
            ret = self.stream.set(4,resolution[1])
                
            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
            self.stopped = False

        def start(self):
        # Start the thread that reads frames from the video stream
            Thread(target=self.update,args=()).start()
            return self

        def update(self):
            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

        def read(self):
        # Return the most recent frame
            return self.frame

        def stop(self):
        # Indicate that the camera and thread should be stopped
            self.stopped = True

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    # Import TensorFlow libraries
    # If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tensorflow')
    if pkg is None:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    while True:
        flag=0
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                cv2.circle(frame, (xmin,ymin), 5, (255, 255, 0), cv2.FILLED)
                cv2.circle(frame, (xmax,ymax), 5, (0, 255, 255), cv2.FILLED)
                x_diff=xmax - xmin
                y_diff=ymax - ymin
                mid_x=x_diff/2 + xmin
                mid_x=math.ceil(mid_x)
                mid_y=ymin+y_diff/2
                mid_y=math.ceil(mid_y)
#                 print("xmin=%d" % xmin)
#                 print("ymin=%d" % ymin)
#                 print("xmax=%d" % xmax)
#                 print("ymax=%d" % ymax)
#                 print("mid_x=%d" % mid_x)
#                 print("mid_y=%d" % mid_y)
                cv2.circle(frame, (0,0), 5, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (mid_x,mid_y), 5, (255, 255, 255), cv2.FILLED)
                 
                
                #cv2.circle(frame, (math.ceil(w/2),math.ceil(h/2)), 7, (255,255,255), cv2.FILLED)
#                 print("h=%d" % math.ceil(h/2))
#                 print("w=%d" % math.ceil(w/2))
#                 if mid_x>w/2:
#                     print("right")
#                 else:
#                     print("left")
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                label_out.append(label)
                mid_x_out.append(mid_x)
                mid_y_out.append(mid_y)
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
#         if cv2.waitKey(1)== ord('q'):
#             break
        #cv2.waitKey(1000)
        #print(label_out)
        #print(mid_x_out)
        #print(mid_y_out)
        (h,w)=frame.shape[:2]
#         for i in label_out:
#             print(i)
#             if "cup" in i:
#                  while(mid_x_out[0]!=w/2):  
#                     if mid_x_out[0]>w/2:
#                         print("object is to the right side of screen")
#                         dist=mid_x_out[0]-w/2
#                         Fright.backward() #acually fleft
#                         Bright.forward()
#                         Fleft.forward() #actually fright
#                         Bleft.backward()
#                         t1=dist*0.05
#                         time.sleep(t1)
#                         Fright.stop()
#                         Bleft.stop()
#                         Fleft.stop()
#                         Bright.stop()
#                     else:
#                             print("obj is to the left side of frame")
#                             dist=mid_x_out[0]-w/2
#                             Fright.forward() #acually fleft
#                             Bright.backward()
#                             Fleft.backward() #actually fright
#                             Bleft.forward()
#                             t1=dist*0.05
#                             time.sleep(t1)
#                             Fright.stop()
#                             Bleft.stop()
#                             Fleft.stop()
#                             Bright.stop()
                        #flag=1
             #move_forward()
             #scan(speech_input)
        if cv2.waitKey(1)== ord('q'):
            break
        #if(flag==1):
            
            #break
    
    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    return()
def speech_recognize():
    r=sr.Recognizer()
    mic_list=sr.Microphone.list_microphone_names()
    for i,microphone_name in enumerate(mic_list):
        #print(microphone_name)
        if microphone_name==mic_name:
            device_id=i
    with sr.Microphone(device_index=device_id,sample_rate=sample_rate,chunk_size=chunk_size)as source:
        r.adjust_for_ambient_noise(source)
        print("say something")
        audio=r.listen(source)
        
        try:
            text=r.recognize_google(audio)
            print("you said: " +text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition sevice:{0}")
    return text
    
def scan(speech_input):
    
    label1,mid_x1,mid_y1,h,w=object_detection()
    print(label1)
    print(mid_x1)
    print(mid_y1)
    object_detection_loop()
    for i in label1:
        if "cup" in i or "toilet" in i:
             adjust_angle(mid_x1,mid_y1,h,w)
             move_forward()
             scan(speech_input)
             #break
        else:
            return()
def scan_360():
    flag=0
    label1,mid_x1,mid_y1,h,w=object_detection()
    print(label1)
    object_detection_loop()
    for i in label1:
        if "cup" in i or "toilet" in i:
            flag=1
    if flag==1:
        return()
    else:
        Fright.forward() #acually fleft
        Bright.backward()
        Fleft.backward() #actually fright
        Bleft.forward()
        #t1=dist*0.05
        time.sleep(0.25)
        Fright.stop()
        Bleft.stop()
        Fleft.stop()
        Bright.stop()
        scan_360()
def adjust_angle(mid_x1,mid_y1,h,w):
  print("mid_x1",mid_x1)
  print("w",w)
  while(mid_x1[0]!=w):  
    if mid_x1[0]>w:
        print("object is to the right side of screen")
        right1(mid_x1[0]-w)
    else:
        print("obj is to the left side of frame")
        left1(mid_x1[0]-w)
    break    
def right1(dist):
    print("diff",dist)
    if(abs(dist)>250):
        Fright.forward() #acually fleft
        Bright.backward()
        Fleft.backward() #actually fright
        Bleft.forward()
        #t1=dist*0.05
        time.sleep(0.25)
        Fright.stop()
        Bleft.stop()
        Fleft.stop()
        Bright.stop()
def left1(dist):
    print("diff",dist)
    if(abs(dist)>250):
        Fright.backward() #acually fleft
        Bright.forward()
        Fleft.forward() #actually fright
        Bleft.backward()
        #t1=dist*0.05
        time.sleep(0.25)
        Fright.stop()
        Bleft.stop()
        Fleft.stop()
        Bright.stop()
def move_forward():
     Fright.forward()
     Fleft.forward()
     Bleft.forward()
     Bright.forward()
     time.sleep(0.7)
     Fright.stop()
     Bleft.stop()
     Fleft.stop()
     Bright.stop()
def move_small_forward():
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
    time.sleep(0.75)
    Fright.stop()
    Bleft.stop()
    Fleft.stop()
    Bright.stop()
# def rotate_360():
#     Fright.forward() #acually fleft
#     Bright.backward()
#     Fleft.backward() #actually fright
#     Bleft.forward()
#     time.sleep(2)
    
def main():        
    #txt=speech_recognize()
    #print(txt)
    #txt1=speech_recognize()
    #print(txt)
    scan_360()
    scan("cup")
    #move_forward()
    pwm=GPIO.PWM(21,100)
    pwm.start(5)
    angle1=10
    duty1=float(angle1)/10 + 2.5
    angle2=160
    duty2=float(angle2)/10 + 2.5
    ck=0
    while ck<=1:
        pwm.ChangeDutyCycle(duty2)
        time.sleep(0.8)
        #move_forward()
        pwm.ChangeDutyCycle(duty1)
        time.sleep(0.8)
        if ck == 0:
            move_small_forward()
        pwm.ChangeDutyCycle(duty2)
        time.sleep(0.8)
        ck=ck+1
    time.sleep(1)
    move_backward()
    #scan_360()
    
#     while(1):
#         label1,mid_x1,mid_y1,h,w=object_detection()
main()
#Bus 001 Device 006: ID 0d8c:013c C-Media Electronics, Inc. CM108 Audio Controller