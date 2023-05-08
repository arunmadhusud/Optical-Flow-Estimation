
# Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach

Optical flow estimation has found applications in various computer vision applications like object detection and tracking, movement detection, robot navigation and visual odometry. 
This project presents a comprehensive study comparing the performance of both the classical and deep learning approaches for estimating dense  optical flow. We used the Farneback method as a representative of classical techniques and FlowNet2 as a representative of deep learning-based methods. 

Our experimental results highlight performance comparison of both the methods on a defined dataset using appropriate metrics - L1 error, Average end point error and Average angular error. The results show that FlowNet 2.0 provides significantly better results than Farneback Algorithm. 

This comparative analysis provides valuable insights for researchers and practitioners seeking to adopt a suitable optical flow estimation technique for their specific applications.

![Workflow](https://i.imgur.com/Y94iPKT.png)

## Executables

Run FlowNet_2_0.ipynb to train, validate and test FlowNet 2.0. Replace main.py and losses.py provided by the authors of FlowNet 2.0 with the ones in this repository in PRCVOpticalFlow/scripts/.

Run ClassicalCVOpticalFlow to either take a prerecorded video or stream from webcam to generate optical flow using Farneback algorithm and track the face based on the generated flow.

Run DeepLearningCVOpticalFlow to iterate through a directory of generated (.flo) files, to import the optical flow fields, and track the face based on the imported flow.

Run ImgtoFlow.py to input sequence of frames and calculate optical flow using the Farneback algorithm. This script saves the generated optical flow as a series of (.flo) files.

Run VisualizeFlow.py to convert sequence of (.flo) files to a sequence of (.png) files.

Run ImgtoVid.py to convert sequence of (.png) images to a video (.mp4).

Run LossesforClassicalCV.py to calculate L1 error, Average Endpoint Error and Average Angular Error for the (.flo) files estimated by the Farneback Algorithm. These metrics are saved to a (.csv) file.

Run change_names_test.py and separate_images.py files to generate custom dataset from the MPI Sintel Flow Dataset.

## Demo



Face Tracking Using Optical Flow: https://youtu.be/YQDdv9CqYyA

Classical Computer Vision Optical Flow: https://youtu.be/aHbXsap42-c

Deep Learning Optical Flow: https://youtu.be/kpLF2yQc9fU

Classical Computer Vision Optical Flow UI: https://youtu.be/nelAh5yYat4

Optical Flow Results Comparison: https://youtu.be/S2ito1zSUEk


## Acknowledgments

Parts of this code were derived, as noted in the code, from [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) and [Gauravv97](https://github.com/Gauravv97/flownet2-pytorch).

Visualization functions for (.flo) files were derived from [flowlib.py](https://github.com/sampepose/flownet2-tf/blob/master/src/flowlib.py)




