import cv2
import numpy as np
import os


CAPTURE  = cv2.VideoCapture(1)
face_knowns_folder = os.path.join(os.getcwd(),'known_faces')
print(face_knowns_folder)

name = input('Please enter name: ')
for i in range(10):
    _,frame = CAPTURE.read()
    show_frame = frame.copy()
    cv2.putText(show_frame,'Face no {}'.format(i+1),(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
    cv2.imshow('Face',show_frame)
    cv2.waitKey(1000)
    cv2.imwrite(os.path.join(face_knowns_folder,'{}_face_{}.jpg'.format(name,i+1)),frame)
         
cv2.destroyAllWindows()
CAPTURE.release()