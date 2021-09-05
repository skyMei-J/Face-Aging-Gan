
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from . import pretrained_networks as pn

import lpips as util


class PErcept(nn.Module,):
    def __init__(self):
        super(PErcept, self).__init__()

        self.scaling_layer = Normal()

        net_type = pn.vgg16
        self.test_loss = nn.L1Loss()
        self.net = net_type(pretrained=True, requires_grad=False)
        self.modulate = 'modulate'
        self.lin0 = Link(64 )
        self.lin1 = Link(128)
        self.lin2 = Link(256)
        self.lin3 = Link(512)
        self.lin4 = Link(512)
        
    def forward(self, in0, in1):

        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        feats0[0], feats1[0] = util.normalize_tensor(outs0[0]), util.normalize_tensor(outs1[0])
        diffs[0] = (feats0[0]-feats1[0])**2
        temp1 = (feats0[0]-feats1[0])
        feats0[1], feats1[1] = util.normalize_tensor(outs0[1]), util.normalize_tensor(outs1[1])
        temp2 = feats1[1]
        #print(temp2)
        diffs[1] = (feats0[1]-feats1[1])**2
        temp3 = (feats0[1]-feats1[1])
        feats0[2], feats1[2] = util.normalize_tensor(outs0[2]), util.normalize_tensor(outs1[2])
        temp2 = feats1[2]
        #print(feats1[2])
        diffs[2] = (feats0[2]-feats1[2])**2
        feats0[3], feats1[3] = util.normalize_tensor(outs0[3]), util.normalize_tensor(outs1[3])
        temp5 = outs1[3]
       # print(temp5)
        diffs[3] = (feats0[3]-feats1[3])**2
        feats0[4], feats1[4] = util.normalize_tensor(outs0[4]), util.normalize_tensor(outs1[4])
        diffs[4] = (feats0[4]-feats1[4])**2

        res = []

        res.append(self.lin0.model(diffs[0]).mean([2,3],keepdim=True))
        res.append(self.lin1.model(diffs[1]).mean([2,3],keepdim=True)    )
        res.append(self.lin2.model(diffs[2]).mean([2,3],keepdim=True)    )
        res.append(self.lin3.model(diffs[3]).mean([2,3],keepdim=True)    )
        res.append(self.lin4.model(diffs[4]).mean([2,3],keepdim=True)    )
        
        test_list = []
        count_list = []
        
        for jj, layer in enumerate(res):
#             print(layer)
            test_list.append(layer)
            count_list.append(jj)
#         print(res[3])
        return res[0]+res[1]+res[2]+res[3]+res[4]

class Normal(nn.Module):
    def __init__(self):
        super(Normal, self).__init__()
        self.normal = 1e5
        tmp1 = torch.Tensor([-.030,-.088,-.188])[None,:,None,None]
        tmp2 = torch.Tensor([.458,.448,.450])[None,:,None,None]
        self.register_buffer('shift',tmp1)
        self.register_buffer('scale', tmp2)

    def forward(self, input):
        tmp1 = input - self.shift
        tmp2 = tmp1/self.scale
        return tmp2


class Link(nn.Module):

    def __init__(self, chn_in):
        super(Link, self).__init__()
        self.layerL=[]
        self.layerL.append(nn.Dropout())
        layers = []
        layers.append(nn.Dropout())
        layers += [nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

