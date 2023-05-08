'''
Authors: Arun Madhusudhanan, Tejaswini Deore

Project for CS 5330 Pattern Recogn & Comput Vision

Title: Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach

This program takes an input sequence of frames and calculates optical flow using the Farneback algorithm. This script saves the generated optical flow as a series of (.flo) files.

'''

# import necessary packages
import cv2
import numpy as np
import os

# define input and output directories
input_dir = "/home/tejaswini/PRCV/FinalProject/scripts/test/final"
output_dir = "/home/tejaswini/PRCV/FinalProject/FlowFiles_scripts"

# create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# loop through each folder inside input directory
for folder_name in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  # skip files that are not directories

    # create corresponding folder inside output directory
    output_folder_path = os.path.join(output_dir, folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # get list of image files in current folder
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")])

    # initialize previous frame and output file
    prev_frame = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    flow_filename = os.path.join(output_folder_path, "flow_0001.flo")
    flow_file = open(flow_filename, 'wb')

    for i in range(1, len(image_files)):
        # read current frame and convert to grayscale
        curr_frame = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)

        # calculate optical flow between previous and current frame
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # write flow to output file
        flow_height, flow_width, _ = flow.shape
        flow_file.write(np.array([80, 73, 69, 72], np.uint8).tobytes())  # write header
        flow_file.write(np.array([flow_width, flow_height], np.int32).tobytes())  # write size
        flow_file.write(flow.tobytes())  # write data

        # update previous frame and output filename
        prev_frame = curr_frame
        flow_filename = os.path.join(output_folder_path, f"flow_{i+1:04}.flo")
        flow_file.close()
        flow_file = open(flow_filename, 'wb')

    flow_file.close()
