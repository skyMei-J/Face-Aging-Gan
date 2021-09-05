
from __future__ import absolute_import
import numpy as np
import torch
from torch import nn
import os
from .base_model import BaseModel
from . import networks_basic as networks
import inspect

class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self,use_gpu=True):
        
        BaseModel.initialize(self)

        self.net = 'vgg'
        self.model_name = '%s [%s]'%('net-lin','vgg')
        self.upsample = 'upsample'
        self.Closs = 1e10 **2
        self.net = networks.PErcept()
        self.test_loss = torch.nn.L1Loss()
        kw = {}
        
        model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize), '..', 'weights/v%s/%s.pth'%('0.1','vgg')))
        
        print('Loading model from: %s'%model_path)
        self.net.load_state_dict(torch.load(model_path, **kw), strict=False)
        self.GPU = True
        self.parameters = list(self.net.parameters())
        self.model = False
        
        self.net.eval()
        self.net.to(0)
        
        self.net = torch.nn.DataParallel(self.net, device_ids=[0])

    def forward(self, in0, in1, retPerLayer=False):
#         print(in0,in1)
        return self.net.forward(in0, in1)

