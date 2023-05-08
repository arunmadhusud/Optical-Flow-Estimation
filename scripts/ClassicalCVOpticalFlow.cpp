//
//  Classical Computer Vision.cpp
//  Title: Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach

//  Project for CS 5330 Pattern Recogn & Computer Vision
//  Run ClassicalCVOpticalFlow to either take a prerecorded video or stream from webcam to generate optical flow using Farneback algorithm and track the face based on the generated flow.

//  Created by Hardik Devrangadi and Francis Jacob Kalliath on 4/22/23.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

int step = 16;
int minStep = 10;
int maxStep = 64;

// Define a global ROI - Region of Interest
Rect roi;

// Function to return the ROI Bounding box of the detected face
Rect returnFace(Mat frame1)
{
    int height=0, width=0,x=0,y =0;
    
    // Create a CascadeClassifier object to load the pre-trained classifier
    cv::CascadeClassifier face_cascade;
    face_cascade.load("/Users/hardikdevrangadi/Desktop/FinalProject/xml/haarcascade_frontalface_alt.xml");
    
    // Detect faces in the image
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(frame1, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    
    // Draw a bounding box around each detected face and return its ROI
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(frame1, faces[i], cv::Scalar(0, 0, 255), 2);
        cv::Mat roi = frame1(faces[i]);
        x = faces[i].x;
        y = faces[i].y;
        height = faces[i].height;
        width = faces[i].width;
    }
    
    Rect roi(x,y,height,width);
    
    return roi;
}

// Void function to create a trackbar (UI)
void on_trackbar(int, void*)
{
    
}

// Void function to create a clickable button (UI)
void onMouse(int event, int x, int y, int flags, void* frame)
{
    if (event == EVENT_LBUTTONDOWN) {
        // Check if the mouse click is within the button rectangle
        Rect buttonRect(750, 10, 200, 60);
        if (buttonRect.contains(Point(x, y))) {
            
            Mat* frame1 = (Mat*)frame;
            roi = returnFace(*frame1);
        }
    }
}


// Main method begins
int main()
{
    // Use this block of code to use live webcam video
    VideoCapture capture(0);
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open webcam!" << endl;
        return 0;
    }
    // Use thisblock of code to use a prerecorded video
    
    //    VideoCapture capture("/Users/hardikdevrangadi/Desktop/FinalProject/videofinal.mp4"); // replace with the path to your video file
    //    if (!capture.isOpened()){
    //        //error in opening the video input
    //        cerr << "Unable to open video file!" << endl;
    //        return 0;
    //    }
    
    // Change the fps, codec and output video size. Make sure video size matches the input
    VideoWriter writer("/Users/hardikdevrangadi/Desktop/FinalProject/classicalfinallive.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(640, 480));
    
    // Capture the first frame
    Mat frame1, prvs;
    capture >> frame1;
    resize(frame1, frame1, Size(), 0.5, 0.5);
    cvtColor(frame1, prvs, COLOR_BGR2GRAY);
    
    int fps = 0;
    double t = 0;
    
    // Create an output window
    namedWindow("Optical Flow", WINDOW_AUTOSIZE);
    
    // Create a Trackbar to vary the stepSize
    createTrackbar("Step Size", "Optical Flow", &step, maxStep, on_trackbar);
    setTrackbarMin("Step Size", "Optical Flow", minStep);
    
    // Returns the coordinates of the ROI box, bounding the detected face in the first frame
    roi = returnFace(frame1);
    
    //Save the flow data in a csv
    ofstream outfile("/Users/hardikdevrangadi/Desktop/FinalProject/flow_values.csv");
    
    while(true){
        Mat frame2, next;
        capture >> frame2;
        resize(frame2, frame2, Size(), 0.5, 0.5);
        if (frame2.empty())
            break;
        cvtColor(frame2, next, COLOR_BGR2GRAY);
        
        Mat flow(prvs.size(), CV_32FC2);
        calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        
        // draw flow field as a grid of arrows
        Mat flow_img = frame2.clone();
        for (int y = 0; y < flow_img.rows; y += step){
            for (int x = 0; x < flow_img.cols; x += step){
                const Point2f& fxy = flow.at<Point2f>(y, x);
                //                cout << fxy.x <<" "<<fxy.y<<endl;
                outfile << fxy.x << "," << fxy.y << endl;
                line(flow_img, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(0, 255, 0), 1, LINE_AA);
                circle(flow_img, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, Scalar(0, 0, 255), -1, LINE_AA);
            }
        }
        
        setMouseCallback("Optical Flow", onMouse, (void*)&frame2);
        
        // calculate average flow vector within ROI
        Mat flow_roi = flow(roi);
        Scalar mean_flow = mean(flow_roi);
        Point2f mean_flow_vec(mean_flow[0], mean_flow[1]);
        
        // calculate and display the velocity
        float velocity = norm(mean_flow_vec);
        putText(flow_img, format("Velocity: %.2f pixels/frame", velocity), Point(30, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        
        String direction;
        if(velocity == 0){
            direction = "Stationary";
        }
        else if (mean_flow_vec.x > 0) {
            direction += "from left to right";
        } else if(mean_flow_vec.x < 0) {
            direction += "from right to left";
        }
        // display object direction
        putText(flow_img, format("Object is moving %s", direction.c_str()), Point(30, 90), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        
        // calculate and display the fps
        double t_prev = t;
        t = (double)cv::getTickCount() / cv::getTickFrequency();
        fps = cvRound(1.0 / (t - t_prev));
        putText(flow_img, format("FPS: %d", fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        
        // update ROI based on flow
        roi.x += mean_flow_vec.x * 0.4 + (0.55+ mean_flow_vec.x);
        roi.y += mean_flow_vec.y * 0.4 + (0.55+ mean_flow_vec.y);
        
        // make sure the ROI stays within the frame
        roi.x = max(0, roi.x);
        roi.y = max(0, roi.y);
        roi.width = min(frame2.cols - roi.x, roi.width);
        roi.height = min(frame2.rows - roi.y, roi.height);
        
        // draw ROI rectangle
        rectangle(flow_img, roi, Scalar(0, 255, 0), 2);
        
        Rect roiface;
        
        // Draw the ground Truth box
        roiface = returnFace(frame2);
        rectangle(flow_img, roiface, Scalar(0, 0, 255), 2);
        
        // Draw the Reset ROI button
        Rect buttonRect(750, 10, 200, 60);
        rectangle(flow_img, buttonRect, Scalar(100, 100, 100), -1);
        
        // Calculate the intersection area of the two boxes
        cv::Rect intersection = roi & roiface;
        if (intersection.width > 0 && intersection.height > 0) {
            vector<Point> intersection_pts;
            
            // Add the four corners of the intersection rectangle
            intersection_pts.push_back(Point(intersection.x, intersection.y));
            intersection_pts.push_back(Point(intersection.x + intersection.width, intersection.y));
            intersection_pts.push_back(Point(intersection.x + intersection.width, intersection.y + intersection.height));
            intersection_pts.push_back(Point(intersection.x, intersection.y + intersection.height));
            
            Mat mask = Mat::zeros(flow_img.size(), flow_img.type());
            fillConvexPoly(mask, intersection_pts, Scalar(255, 0, 0
                                                          ));
            
            Mat masked_flow_img;
            bitwise_and(flow_img, mask, masked_flow_img); // Apply mask to the flow_img
            
            double opacity = 0.75;
            addWeighted(masked_flow_img, opacity, mask, 1 - opacity, 0, masked_flow_img); // Apply opacity to the masked image
            
            double intersection_area = intersection.width * intersection.height;
            double smaller_rect_area = min(roi.area(), roiface.area());
            
            double intersection_percent = intersection_area / smaller_rect_area * 100.0;
            
            string text = "Intersection: " + std::to_string(intersection_percent) + "%";
            text = text.substr(0, text.find(".") + 3); // Truncate to two decimal places
            cv::putText(flow_img, text, Point(intersection.x, intersection.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            
            // Copy the masked image back to the original flow_img
            masked_flow_img.copyTo(flow_img, mask);
        }
        
        
        // Draw the text
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 1;
        int thickness = 1;
        string text = "Reset ROI";
        Size textSize = getTextSize(text, fontFace, fontScale, thickness, NULL);
        Point textOrg(buttonRect.x + (buttonRect.width - textSize.width) / 2,
                      buttonRect.y + (buttonRect.height + textSize.height) / 2);
        putText(flow_img, text, textOrg, fontFace, fontScale, Scalar(255, 255, 255), thickness);
        
        //Display the output window
        imshow("Optical Flow", flow_img);
        
        //Write it to a video
        writer.write(flow_img);
        
        int keyboard = waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;
        
        prvs = next;
    }
    outfile.close();
    return 0;
}
