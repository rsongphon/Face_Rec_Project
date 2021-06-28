import face_recognition
import cv2
import os


CWD_DIR = os.getcwd()
KNOWN_FACE_DIR = os.path.join(CWD_DIR,'known_faces')
UNKNOW_FACE_DIR = os.path.join(CWD_DIR,'unknown_faces')

#OpenCV Properties of rectangle to label the face
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_LINE = cv2.LINE_AA
FRAME_THICKNESS = 2
FONT_THICKNESS = 2

# Machice learning model
MODEL = 'cnn'

# More tolerance = more detection but chance for error (False Positive)
TOLERANCE = 0.6

## walk in know face directory
# load every image in face directory
# encode image