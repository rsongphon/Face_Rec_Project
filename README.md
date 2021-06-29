# Face_Rec_Project #


### First implementation ### 
Using Face Recognition package
https://pypi.org/project/face-recognition/

# Concept #
https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
https://pythonprogramming.net/facial-recognition-python/

- Using Deep learning (CNN and HOG)

- If use CNN (Faster) it is best pratice to use with **CUDA** otherwise it will get really slow
(Deep learning use heavy computing process so use GPU is a must)

-  Pre-trained network : The network has already been trained to create 128-d embeddings. no need to train network 
# Requirement #
1. dlib   : contains our implementation of “deep metric learning” which is used to construct our face embeddings used for the actual recognition process.
2. face_recognition library : wraps around dlib’s facial recognition functionality, making it easier to work with.
3. OpenCV for image manipulation
4. CUDA (use with cnn model) from NVIDIA
5. CUDA cuDNN : GPU Accelerated library for deep neural network
6. Tensorflow ?
7. virtual enviroment : in case things go messy

# Installation (Windows only) #
doing this in order
1. Install CUDA : https://www.skconan.com/cuda-windows-10/
2. Insall CUDA cuDNN
3. Install tesorflow
4. Create virtual enviroment
5. For window only : https://github.com/ageitgey/face_recognition/issues/175
    5.1 Install Microsoft Visual Studio 2015 (or newer) with C/C++ Compiler installed
    5.2 CMake for windows and add it to your system environment variables.
6. Install dlib via pip
7. install face_regcogniton via pip
8. install other library (opencv, etc)

# Problem #
1. dataset need time to re-encoding when running script (take time) 
    Solution : dump encoding data into binary file using pickle module (no need to encoding face every time)
    https://arnondora.in.th/how-to-pickle-and-unpickle-python/ 

2. Raspberry Pi limitation:
    2.1 The Raspberry Pi does not have enough memory to utilize the more accurate CNN-based face detecto
    2.2 limited to HOG instead except that HOG is far too slow on the Pi for real-time face detection
    2.3 utilize OpenCV’s Haar cascades instead


# Practice Specification #
- A program is use for login staff (authentication) in companay
- A staff stand in front of the camera to log in
- a camera detect person from pre training network

IDEA
- A person can be log in one staff at a time
- if more than 1 person appear in the frame >>> no face regcognize . Just detect the face 
- Regcongnize the face and send data to server to stamp time and log-in
- Have an algorithm to seperate morning login, noon logout (or server side do this?)
- Program can check if person is already login in the range of given time (morning login, noon logout) : 
    - Using API to get data from main server ?
- Have a screen to show staff name , time after authentication
- Have dataset encoding prepare in advance as a file to load (encoding using powerful computer) to reduce processing time
- showing unknow face detect in the screen
- (optinal) Have detect guest mode to detect unknow face and snapshot that face. use as and input for dataset to training and assign id
    to remember that person and have reference to detect that person later
- If staff face have many condition (glass , hat , scar , makeup , ETC). We need more data for this person. 
  In this case a program must have mode to manually capture face image and send data to the server use for training later   

