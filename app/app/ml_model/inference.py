import torch
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import scipy.misc
import numpy as np
import math
from .network import Net
from PIL import Image
import sys



class MyModel:

    def __init__(self):
        self.net = Net(size=256)
        self.net = nn.DataParallel(self.net)
        self._initialize()


    def _initialize(self):
        # Load weights
        print(sys.path)
        print('Loading Model')
        # state_dict = torch.load('/Users/ishangupta/Documents/flask-can/app/app/ml_model/trained_weights.pth')
        state_dict = torch.load('app/ml_model/weights/trained_weights.pth')
        self.net.load_state_dict(state_dict)



    def scale_image(self,image):
        image -= image.min()
        image /= image.max()
        image *= 255
        return image.astype(np.uint8)

    def run(self):

        # Generate latent vector
        x = torch.randn(1, 100, 1, 1)
        x = Variable(x, volatile=True)

        print('Executing forward pass')
        images = self.net(x)


        images_np = images.data.numpy().transpose(0, 2, 3, 1)
        image_np = self.scale_image(images_np[0, ...])

        im = Image.fromarray(image_np)
        print('Saving Image.....')
        return im
        #im.save("/Users/ishangupta/Documents/flask-can/app/app/static/img/generated.jpg")



