# Face_Rec_Project


### First implementation ### 
Using Face Recognition package
https://pypi.org/project/face-recognition/

# Concept #
https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
https://pythonprogramming.net/facial-recognition-python/

Using Deep learning (CNN and HOG)

If use CNN (Faster) it is best pratice to use with **CUDA** otherwise it will get really slow
(Deep learning use heavy computing process so use GPU is a must)

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

