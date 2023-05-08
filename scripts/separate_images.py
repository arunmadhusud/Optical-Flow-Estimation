'''
Authors: Arun Madhusudhanan, Tejaswini Deore

Project for CS 5330 Pattern Recogn & Comput Vision

Title: Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach

Run to generate custom dataset from the MPI Sintel Flow Dataset.
'''

import os
import shutil

# set the root directory where the folders containing the images are
root_dir = '/home/marley/test_scripts/MPI-Sintel-complete/training'

# set the percentage of images to be used for the training set
train_percent = 0.7

train_dir = '/home/marley/test_scripts/MPI-Sintel-complete/scripts/train'
test_dir = '/home/marley/test_scripts/MPI-Sintel-complete/scripts/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


print(root_dir)
print("root dir: ", os.listdir(root_dir))


check = ['flow',  'clean', 'final']

# loop over all the folders in the root directory
for folder in os.listdir(root_dir):

    if folder in check:

        print("folder: ", folder)
        folder_p = os.path.join(root_dir, folder)
        


        print(os.path.join(train_dir, folder))

        os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

        for subfolder in os.listdir(os.path.join(root_dir, folder)):


            print("subfolder: ", subfolder)

            # create a full path to the current folder
            folder_path = os.path.join(folder_p, subfolder)

            # print("folder path: ", folder_path)

            # check if the current path is a directory
            if os.path.isdir(folder_path):
                # create a subdirectory for the train and test sets
                # train_dir = os.path.join(folder_path, 'train_')
                # test_dir = os.path.join(folder_path, 'test')
                # os.makedirs(train_dir, exist_ok=True)
                # os.makedirs(test_dir, exist_ok=True)

                dir_test_new = os.path.join(test_dir, folder)
                dir_test = os.path.join(dir_test_new, subfolder)


                dir_train_new = os.path.join(train_dir, folder)

                dir_train = os.path.join(dir_train_new, subfolder)

                print("dir_test: ", dir_test)
                print("dir_train: ", dir_train)

                os.makedirs(dir_test, exist_ok=True)
                os.makedirs(dir_train, exist_ok=True)


                # get a list of all the files in the current folder
                files = os.listdir(folder_path)
                files.sort()

                # create a list of only image files
                images = [file for file in files if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') or  file.endswith('.flo')]

                # calculate the number of images to be used for the training set
                num_train = int(len(images) * train_percent)

                num_test = len(images) - num_train

                if folder == 'flow':
                    num_train = int(len(images) * train_percent) 
                    num_test = len(images) - num_train 


                # copy the first num_train images to the train set directory
                for img in images[:num_train]:
                    src = os.path.join(folder_path, img)
                    # print(src)
                    dst = os.path.join(dir_train, img)
                    shutil.copy(src, dst)

                # copy the remaining images to the test set directory
                for img in images[num_test:]:
                    src = os.path.join(folder_path, img)
                    dst = os.path.join(dir_test, img)
                    shutil.copy(src, dst)
