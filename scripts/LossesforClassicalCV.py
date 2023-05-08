'''
Authors: Arun Madhusudhanan, Tejaswini Deore

Project for CS 5330 Pattern Recogn & Comput Vision

Title: Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach

Run to calculate L1 error, Average Endpoint Error and Average Angular Error for the (.flo) files estimated by the Farneback Algorithm. These metrics are saved to a (.csv) file.
'''

# import necessary packages
import torch
import torch.nn as nn
import math
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import math
import csv

def read_flow(filename):
    '''
    Method to read optical flow from flow file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    '''
    # Open the flow file in binary mode
    with open(filename, 'rb') as f:

        # Read the magic number from the file and check if it is valid
        magic = torch.tensor(np.fromfile(f, np.float32, count=1))[0]
        assert magic == 202021.25, 'Flow number %r incorrect. Invalid .flo file' % magic

        # Read the height and width of the flow matrix from the file
        h = int(torch.tensor(np.fromfile(f, np.int32, count=1))[0])
        w = int(torch.tensor(np.fromfile(f, np.int32, count=1))[0])

        # Read the optical flow data from the file and convert it into a matrix
        data = torch.tensor(np.fromfile(f, np.float32, count=int(2*w*h)))
        flow = data.reshape((h, w, 2))

        # print(flow.shape)
    # Return the flow matrix
    return flow

def EPE(input_flow, target_flow):
    '''Method to calculate the end point error between the predicted and target flow
    Args:
        input_flow: predicted flow
        target_flow: ground truth flow
    Returns:
        End point error between the predicted and target flow
    '''
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

def L1(output, target):
    ''' Method to calculate the L1 loss between the predicted and target flow
    Args:   
        output: predicted flow
        target: ground truth flow
    Returns:
        L1 loss between the predicted and target flow
    '''
    return torch.abs(output - target).mean()

class FlowStats(nn.Module):
    
    def _init_(self):
        super(FlowStats, self)._init_()
        self.loss_labels = ['L1', 'EPE','AE']
    
    def get_angular_error(self, pred_flow, true_flow):
        """Calculates the angular error between two flow fields.

        Args:
            pred_flow: The predicted flow field.
            true_flow: The ground truth flow field.

        Returns:
            The angular error between the two flow fields in degrees.
        """
        
        # Calculate the magnitude of the two flow fields.
        pred_mag = torch.norm(pred_flow,p=2,dim=1)
        true_mag = torch.norm(true_flow,p=2,dim=1)

        # Check if either of the magnitude tensors is zero
        zero_mag = (pred_mag == 0) | (true_mag == 0)

        # Calculate the dot product of the two flow fields.
        dot_product = torch.sum(pred_flow * true_flow, dim=1)

        # Calculate the angular error in radians.
        angular_error_radians = torch.acos(torch.clamp(dot_product / (pred_mag * true_mag), -1, 1))

        # Set the angular error to zero for tensors with zero magnitude
        angular_error_radians[zero_mag] = 0

        # Convert the angular error to degrees.
        angular_error_degrees = (180 / math.pi) * angular_error_radians

        return angular_error_degrees.mean()

    def forward(self, output, target):
        '''
        Method to get the L1 loss, EPE and angular error between the predicted and target flow
        Args:
            output: predicted flow
            target: ground truth flow   
        Returns:
            L1 loss, EPE and angular error between the predicted and target flow
        '''
        lossvalue = L1(output, target)
        epevalue = EPE(output, target)
        angular_error_degrees = self.get_angular_error(output, target)
        return [lossvalue, epevalue, angular_error_degrees]

# Define the paths to the input and target root folders
input_root_folder = '/home/tejaswini/PRCV/FinalProject/FlowFiles_scripts'
target_root_folder = '/home/tejaswini/PRCV/FinalProject/scripts/test/flow'

# Get a list of all the input folders
input_folders = sorted(os.listdir(input_root_folder))

# Define the output csv file path
output_csv_path = '/home/tejaswini/PRCV/FinalProject/stats.csv'

# Define the headers for the csv file
headers = ['Folder', 'Max_EPE', 'Mean_EPE',
           'Max_L1_error', 'Mean_L1_error',
           'Max_Angular_Error', 'Mean_Angular_Error']

# Initialize the csv file with headers
with open(output_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(headers)

# Loop over the folders and compute the losses
for folder in input_folders:
    # Define the paths to the input and target sub-folders
    input_folder = os.path.join(input_root_folder, folder)
    target_folder = os.path.join(target_root_folder, folder)

    # Get a list of all the input and target file paths
    input_files = sorted(glob.glob(os.path.join(input_folder, '*.flo')))
    target_files = sorted(glob.glob(os.path.join(target_folder, '*.flo')))
    
    # Initialize lists to store errors
    epe_diff_list = []
    l1_error_list = []
    angular_error_list = []

    # Loop over the file pairs and compute the losses
    for input_file, target_file in zip(input_files, target_files):
        # Read the input and target flow fields
        output = read_flow(input_file)
        target = read_flow(target_file)

        # create instance of FlowStats class
        flow_stats = FlowStats()

        # Get the file names
        input_name = os.path.basename(input_file)
        target_name = os.path.basename(target_file)

        # Calculate the errors
        l1_loss, epe_value, angular_error = flow_stats.forward(output, target)

        # Add errors to lists
        epe_diff_list.append(epe_value)
        l1_error_list.append(l1_loss)
        if not np.isnan(angular_error):
            angular_error_list.append(angular_error)

        # print all the errors
        # print(f'\nInput file: {input_name}, Target file: {target_name}')
        print(f"EPE : {epe_value:.4f}")
        print(f'L1 error: {l1_loss:.4f}')
        print(f"angular_diff : {angular_error}")

    # Calculate max and mean of errors

    epe_diff_max = max(epe_diff_list)
    epe_diff_mean = sum(epe_diff_list) / len(epe_diff_list)

    l1_error_max = max(l1_error_list)
    l1_error_mean = sum(l1_error_list) / len(l1_error_list)

    angular_error_max = max(angular_error_list)
    angular_error_mean = sum(angular_error_list) / len(angular_error_list)

    # Print max and mean of errors
    print(f'\nInput folder: {folder}, target folder: {folder}')
    print(f"Max EPE diff: {epe_diff_max}")
    print(f"Mean EPE diff: {epe_diff_mean}\n")

    print(f"Max L1 error: {l1_error_max}")
    print(f"Mean L1 error: {l1_error_mean}\n")

    print(f"Max Angular error: {angular_error_max}")
    print(f"Mean Angular error: {angular_error_mean}\n")

    # Add statistics to csv file
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([folder, epe_diff_max.item(), epe_diff_mean.item(),
                         l1_error_max.item(), l1_error_mean.item(),
                         angular_error_max.item(), angular_error_mean.item()])
