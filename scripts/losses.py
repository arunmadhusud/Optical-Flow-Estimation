'''
Portions of this code copyright 2017, Clement Pinard
'''

'''
Edited by Arun Madhusudhanan and Tejaswini Deore

The actual L1loss class defined by authors contains only L1 error and EPE.
We have added angular error as well. Hence only the L1loss class is modified. and respective comments are added.
Replace the 'losses.py' provided by authors with this file to obtain the angular error as well
'''
'''
Project for CS 5330 Pattern Recogn & Comput Vision

Title: Comparative Analysis of Optical Flow Techniques: Classical Computer Vision vs Deep Learning Approach
'''

import torch
import torch.nn as nn
import math


def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
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

        # Calculate the dot product of the two flow fields.
        dot_product = torch.sum(pred_flow * true_flow, dim=1)

        # Calculate the angular error in radians.
        angular_error_radians = torch.acos(torch.clamp(dot_product / (pred_mag * true_mag), -1, 1))

        # Convert the angular error to degrees.
        angular_error_degrees = (180 / math.pi) * angular_error_radians

        return angular_error_degrees.mean()

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        angular_error_degrees = self.get_angular_error(output, target)
        return [lossvalue, epevalue, angular_error_degrees]
    
    

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*EPE(output_, target_)
                lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]
