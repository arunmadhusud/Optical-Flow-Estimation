'''
Authors: Arun Madhusudhanan, Tejaswini Deore

Project for CS 5330 Pattern Recogn & Comput Vision

Title: Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach

Run to convert sequence of (.png) images to a video (.mp4).

'''

import cv2
import os

# Define the path to the directory containing the images
img_dir = '/home/tejaswini/PRCV/FinalProject/scripts/test/clean/alley_1'

# Get a list of all the image file names in the directory
img_files = sorted([os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)])

# Open the first image to get its dimensions
img = cv2.imread(img_files[0])
height, width, channels = img.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
out = cv2.VideoWriter('alley_1_clean.mp4', fourcc, 2, (width, height))  # 25 fps

# Loop through each image, convert it to a video frame, and write it to the output video file
for img_file in img_files:
    img = cv2.imread(img_file)
    out.write(img)

# Release the VideoWriter and destroy all windows
out.release()
cv2.destroyAllWindows()
