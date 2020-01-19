import torch
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import scipy.misc
import numpy as np
import math
from PIL import Image


class Net(nn.Module):
    def __init__(self,nz=100,output_nc=3,ngf=64,size=256,norm_layer=nn.BatchNorm2d,use_sigmoid=True):
        super(Net,self).__init__()
        kw = 4

        n_layers = int(math.log(size,2)-3)
        ng_mult = 2 ** (n_layers)

        sequence = [
            nn.ConvTranspose2d(nz,ngf*ng_mult,kernel_size=kw,stride=1,padding=0,bias=False),
            nn.ReLU(True)
        ]

        for n in range(0,n_layers):
            ng_mult = 2**(n_layers-1-n)

            sequence += [
                #nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(ngf*ng_mult*2,ngf*ng_mult,
                                   kernel_size=kw,stride=2,padding=1,bias=False),
                norm_layer(ngf*ng_mult),
                nn.ReLU(True)
            ]

        sequence += [
            nn.ConvTranspose2d(ngf,output_nc,kernel_size=kw,stride=2,padding=1,bias=False)
        ]

        if use_sigmoid:
            sequence += [nn.Tanh()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return self.model(input)