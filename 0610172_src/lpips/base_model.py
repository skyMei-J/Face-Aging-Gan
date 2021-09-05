import os
import numpy as np
import torch
from torch.autograd import Variable
from pdb import set_trace as st
from IPython import embed

class BaseModel():
#     def __init__(self):
#         pass;
        
#     def name(self):
#         return 'BaseModel'

    def initialize(self):
        self.use_gpu = True
        self.gpu_ids = gpu_ids=[0]