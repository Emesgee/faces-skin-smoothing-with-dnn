#include <iostream> 
#include <string> 
#include <vector> 
#include <stdlib.h> 
#include <opencv2/core.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/dnn.hpp> // Deep Neural Network (DNN) module of OpenCV

// defining namespaces
using namespace cv;
using namespace std;
using namespace cv::dnn;


const size_t inWidth = 300; // input image width
const size_t inHeight = 300; // input image Height
const double inScaleFactor = 1.0; // scale factor for input image 
const double confidenceThreshold = 0.18; // confidence threshold for face detection 
const cv::Scalar meanVal(104.0, 177.0, 123.0); // mean pixels value for normalization


// path to pre-trained dnn model 
string MODEL_PATH = "../../data/models/";

// tensorflow config file
const std::string tensorflowConfigFile = MODEL_PATH + "opencv_face_detector.pbtxt";
// tensorflow weight file
const std::string tensorflowWeightFile = MODEL_PATH + "opencv_face_detector_uint8.pb";

// detect faces function with OpenCV DNN
void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN)
{
    //  get frame height and width
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    // preprocess input Image 
    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight),
        meanVal, false, false);

    // set input for neural network
    net.setInput(inputBlob, "data");

    // forward pass to get detection
    cv::Mat detection = net.forward("detection_out");

    // get detection matrix
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    // collect the coordinates of all detected faces
    vector<Rect> facesRects;

    // iterate over each detected face
    for (int i = 0; i < detectionMat.rows; i++)
    {
        // get confidence of the detection
        float confidence = detectionMat.at<float>(i, 2);

        // check if confidence is above threshold
        if (confidence > confidenceThreshold)
        {
            // calculate coordinates if above threshold
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            // append detected face rectangle to the list
            facesRects.push_back(Rect(x1, y1, x2 - x1, y2 - y1));

            // Inside the loop where faces are detected
            cout << "Detected Face " << i + 1 << " - (x1,y1): (" << x1 << "," << y1 << "), (x2,y2): (" << x2 << "," << y2 << ")" << endl;
        }
    }

    // convert the entire image to HSV color space
    Mat hsvImg;
    cvtColor(frameOpenCVDNN, hsvImg, COLOR_BGR2HSV);

    // define the lower and upper bounds for skin color in HSV
    Scalar lowerBound = Scalar(0, 40, 80); // Example lower bound for skin color in HSV
    Scalar upperBound = Scalar(20, 255, 255); // Example upper bound for skin color in HSV

    // threshold the HSV image to get a binary mask of the skin regions
    Mat skinMask;
    inRange(hsvImg, lowerBound, upperBound, skinMask);

    // apply a smoothing filter (e.g., Gaussian blur) to the skin regions
    Mat smoothed;
    int neighborhoodPixel = 12; // 12 pixels in diameter

    bilateralFilter(frameOpenCVDNN, smoothed, 10, 20, 20);

    // overlay smoothed face regions onto the original image using the skin mask
    Mat result = frameOpenCVDNN.clone(); // Create a copy of the original image
    for (const Rect& faceRect : facesRects)
    {
        Mat faceROI = smoothed(faceRect); // Extract the face region from the smoothed image
        faceROI.copyTo(result(faceRect), skinMask(faceRect)); // Overlay the smoothed face onto the original image
    }

    // display the result
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Smoothed Faces", result);

    // write image
    imwrite("result.jpg", result);
}

//load pre - trained model
Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);


int main() {

    // intial Text
    cout << "Hi and welcome to my face detection and sin smoothing app: " << endl;

    // print OpenCV version
    cout << "OpenCV version: " << CV_VERSION << endl;

    // read input image
    Mat img = imread("faces.jpg");
   

    // check if the image is loaded successfully
    if (img.empty()) {
        cerr << "Failed to load the image." << endl;
        return -1;
    }

    // create a window and display the image
    namedWindow("Image", WINDOW_NORMAL); // WINDOW_NORMAL allows resizing the window
    imshow("Image", img);
    
        // apply dnn function
    detectFaceOpenCVDNN(net, img);


    // wait for a key press
    waitKey(0);

    cout << "-----------------------------------------------------------" << endl;
    cout << "-----------------------------------------------------------" << endl;
    cout << "-----Image ------------------------------------------------" << endl;
    cout << "-------------------Saved ----------------------------------" << endl;
    cout << "----------------------------------as ----------------------" << endl;
    cout << "-----------------------------------------------result.jpg--" << endl;
    cout << "-----------------------------------------------------------" << endl;
    cout << "-----------------------------------------------------------" << endl;
  

    // close the window
    destroyAllWindows();

    return 0;
}
