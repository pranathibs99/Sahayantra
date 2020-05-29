# Sahayantra
## Major Project - Patient Assistance Robot
Patients in hospitals and homes need constant medical supervision and care. In the present scenario, there is a shortage of nurses in hospitals, and caretakers are too expensive to hire. Added to this is the threat of getting infected from a patient with a communicable disease like COVID-19. This is where robotics could come in handy. Through this project, we propose a patient assistance robot, Sahayantra, which is built to perform elementary tasks like delivering medicines and sending emergency notifications to the nurse or patient's family. For this purpose, we have used a raspberry pi based system that takes speech input from the patient and employs object detection and tracking using TensorFlow lite to find the desired object. Our goal is to provide a personalized and cost-effective robot that is capable of helping the patients in their day to day activities. We have tested the functionalities of Sahayantra against varied environments and with different objects. We enclose our findings in this report.
### Steps for building Sahayantra
1. Install Raspbian on Raspberry Pi
2. Build the robot buggy by attaching DC motors to the Raspberry Pi and l293D motor driver 
guide : https://projects.raspberrypi.org/en/projects/build-a-buggy
3. Program raspberry pi using 'Motor' Library (code in repo : move.py)
4. Install tensorflow lite in a virtual env for object detection
guide: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md 
Sample object detction model is already there in the repo, Use a USB web camera for object detction input
5. Install speech recognition libraries, buy a USB microphone for taking speech recognition input
guide: https://www.geeksforgeeks.org/speech-recognition-in-python-using-google-speech-api/
6. Run the code TFlite_detection_webcam.py for checking object detction.
7. TFlite_detection_webcam_new.py is the complete code used for the project
8. Buy a robotic gripper with a servo motor. and program the servo motor. Run servo1.py in the repo to check the working of the gripper.
