//
//  Deep Learning Computer Vision.cpp
//  Title: Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach

//  Project for CS 5330 Pattern Recogn & Computer Vision
//  Run DeepLearningCVOpticalFlow to iterate through a directory of generated (.flo) files, to import the optical flow fields, and track the face based on the imported flow.
//  Created by Hardik Devrangadi and Francis Jacob Kalliath on 4/22/23.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/optflow.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

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
        //        cv::rectangle(frame1, faces[i], cv::Scalar(0, 0, 255), 2);
        cv::Mat roi = frame1(faces[i]);
        x = faces[i].x;
        y = faces[i].y;
        height = faces[i].height;
        width = faces[i].width;
    }
    
    Rect roi(x,y,height,width);
    
    return roi;
}

// Main method starts
int main()
{
    VideoCapture capture("/Users/hardikdevrangadi/Desktop/FinalProject/videofinal.mp4");
    // replace with the path to your video file
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open video file!" << endl;
        return 0;
    }
    
    Mat frame1, prvs;
    capture >> frame1;
    
    // open video writer
    VideoWriter writer("/Users/hardikdevrangadi/Desktop/FinalProject/deeplearningfinal.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(640, 448)); // change the output file name, codec, FPS and output resolution as per your requirement
    roi = returnFace(frame1);
    // loop through the .flo files
    for (int i = 0; i <=1052 ; i++) { // assuming you have 100 .flo files numbered from 1 to 100
        
        Mat frame2, prvs;
        capture >> frame2;
        
        
        // read optical flow from a .flo file
        Mat flow = readOpticalFlow(format("/Users/hardikdevrangadi/Desktop/FinalProject/videofinal/%06d.flo", i));
        cout << flow.size();
        // display the optical flow as a grid of arrows
        Mat flow_img = Mat::zeros(flow.size(), CV_8UC3);
        int step = 16;
        for (int y = 0; y < flow_img.rows; y += step){
            for (int x = 0; x < flow_img.cols; x += step){
                const Point2f& fxy = flow.at<Point2f>(y, x);
                line(flow_img, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(0, 255, 0), 1, LINE_AA);
                circle(flow_img, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, Scalar(0, 0, 255), -1, LINE_AA);
            }
        }
        
        
        //calculate average flow vector within ROI
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
        //        cv::rectangle(flow_img, cv::Rect(300, 200, 200, 200), cv::Scalar(0, 255, 0), 2);

        
        roi.x += mean_flow_vec.x * 0.2 + (0.55+ mean_flow_vec.x);
        roi.y += mean_flow_vec.y * 0.2 + (0.55+ mean_flow_vec.y);
        
        // make sure the ROI stays within the frame
        roi.x = max(0, roi.x);
        roi.y = max(0, roi.y);
        roi.width = min(frame2.cols - roi.x, roi.width);
        roi.height = min(frame2.rows - roi.y, roi.height);
        rectangle(flow_img, roi, Scalar(0, 255, 0), 2);
        
        Rect roiface;
        
        // draw ROI rectangle
        roiface = returnFace(frame2);
        rectangle(flow_img, roiface, Scalar(0, 0, 255), 2);
        
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
        
        // display frame
        resize(frame2, frame2, Size(640, 448));
        imshow("Optical Flow Video", frame2);
        imshow("Optical Flow", flow_img);
        
        Mat frame_blend;
        double alpha = 0.5; // blending factor
        addWeighted(frame2, alpha, flow_img, 1 - alpha, 0, frame_blend);
        imshow("Optical Flow1", frame_blend);
        
        //Write to a video
        writer.write(frame_blend);
        
        // wait for key press and exit if 'q' is pressed
        int key = waitKey(10); // wait for 10 ms
        if (key == 'q') {
            break;
        }
    }
}


