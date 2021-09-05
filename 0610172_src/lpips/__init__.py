
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.measure import compare_ssim
import torch
from torch.autograd import Variable

from lpips import dist_model

class PerceptualLoss(torch.nn.Module):
    def __init__(self,model='net-lin', net='vgg', use_gpu='cuda'.startswith('cuda')
): 
        super(PerceptualLoss, self).__init__()
        print('========Calculating Perceptual loss====================')
        self.model = dist_model.DistModel()
        self.setup = True
        self.model.initialize(use_gpu='cuda'.startswith("cuda"))
        print('...Done')

    def forward(self, pred, target):
        return self.model.forward(target, pred)

def normalize_tensor(input):
    sq = input**2
    temp = torch.sum(input**2,dim=1,keepdim=True)
#     print(temp)
    norm_factor = torch.sqrt(temp)
    return input/(norm_factor+0.0000000001)

