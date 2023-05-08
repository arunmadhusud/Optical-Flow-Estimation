'''
Authors: Arun Madhusudhanan, Tejaswini Deore

Project for CS 5330 Pattern Recogn & Comput Vision

Title: Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach

Run to update the names of the files inside the custom dataset generated from the MPI Sintel Flow Dataset.
'''


import os

# define the root directory
root_dir = '/home/marley/test_scripts/MPI-Sintel-complete/scripts/test'

check = ['.AppleDouble']

for folder in os.listdir(root_dir):

    if folder not in check:

        folder_p = os.path.join(root_dir, folder)
        
        for subfolder in os.listdir(folder_p):
            subfolder_p = os.path.join(folder_p, subfolder)
            
            i = 1

            # sort all files in the subfolder

            files = os.listdir(subfolder_p)
            files.sort()

            for file in files:
                file_p = os.path.join(subfolder_p, file)
                # print(file_p)

                # change the name of the file with number 00 to 100
                # change the name of the file
                
                # check the extension of the file
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.flo'):

                    # create a new name for the file with the end of the file
                    new_name = 'frame_{:04d}'.format(i) + file[-4:]

                # if :
                #     new_name = 'frame_{:04d}'.format(i) + '.flo'

                # new_name = 'frame_{:04d}'.format(i) + '.png'

                # print(new_name)

                i = i + 1

                # create a new path for the file
                new_path = os.path.join(subfolder_p, new_name)

                print(new_path)

                # rename the file with the new name
                os.rename(file_p, new_path)
