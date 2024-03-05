Face Detection and Skin Smoothing App
-----------------------------------------
This is a C++ application for detecting faces in images 
and applying skin smoothing using OpenCV's Deep Neural Network (DNN) module.

Overview
This application utilizes OpenCV's DNN module to detect faces in an input image. 
It then applies skin smoothing to the detected face regions using a bilateral filter 
and overlays the smoothed faces onto the original image. 
The face detection model and configuration files are provided in the repository.

Requirements
C++ compiler supporting C++11 standard
OpenCV (4.0 or higher) with DNN module
CMake (for building)

Build the executable using CMake and insert the input image in the Release folder.
In order to build the executable please follow these steps:
1: Unzip the file
2: Create a build folder inside the unzipped folder
3: In the command line type: cmake -G "Visual Studio 16 2019" ..
You can choose your own VS version in my case it was:  cmake -G "Visual Studio 17 2022" .. '
4: Now, to run the executable: ..\build\Release\submission.exe

PS: Remember to change the  MODEL_PATH  to the correct path --> examle ../models where both
opencv_face_detector.pbtxt and opencv_face_detector_uint8.pb are located.

Credits
This application was developed by Your Mohammad Ghadban.

Cheers!



# Developer:

Make it more readable

I need to make a better readme file! test

