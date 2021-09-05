from collections import namedtuple
import torch
from torchvision import models as tv
from IPython import embed


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        VGG = tv.vgg16(pretrained=True).features
        
        self.layer1 = torch.nn.Sequential()
        self.layer2 = torch.nn.Sequential()
        self.layer3 = torch.nn.Sequential()
        self.layer4 = torch.nn.Sequential()
        self.layer5 = torch.nn.Sequential()
        self.layer1.add_module(('0'),  VGG[0])
        self.layer1.add_module(('1'),  VGG[1])
        self.layer1.add_module(('2'),  VGG[2])
        self.layer1.add_module(('3'),  VGG[3])
        self.layer2.add_module(('4'),  VGG[4])
        self.layer2.add_module(('5'),  VGG[5])
        self.layer2.add_module(('6'),  VGG[6])
        self.layer2.add_module(('7'),  VGG[7])
        self.layer2.add_module(('8'),  VGG[8])
        self.layer3.add_module(('9' ), VGG[9 ])
        self.layer3.add_module(('10'), VGG[10])
        self.layer3.add_module(('11'), VGG[11])
        self.layer3.add_module(('12'), VGG[12])
        self.layer3.add_module(('13'), VGG[13])
        self.layer3.add_module(('14'), VGG[14])
        self.layer3.add_module(('15'), VGG[15])
        self.layer4.add_module(('16'), VGG[16])
        self.layer4.add_module(('17'), VGG[17])
        self.layer4.add_module(('18'), VGG[18])
        self.layer4.add_module(('19'), VGG[19])
        self.layer4.add_module(('20'), VGG[20])
        self.layer4.add_module(('21'), VGG[21])
        self.layer4.add_module(('22'), VGG[22])
        self.layer5.add_module(('23'), VGG[23])
        self.layer5.add_module(('24'), VGG[24])
        self.layer5.add_module(('25'), VGG[25])
        self.layer5.add_module(('26'), VGG[26])
        self.layer5.add_module(('27'), VGG[27])
        self.layer5.add_module(('28'), VGG[28])
        self.layer5.add_module(('29'), VGG[29])


        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        out = self.layer1(X)
        h_relu1_2 = out
        out = self.layer2(out)
        h_relu2_2 = out
        out = self.layer3(out)
        h_relu3_3 = out
        out = self.layer4(out)
        h_relu4_3 = out
        out = self.layer5(out)
        h_relu5_3 = out
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


