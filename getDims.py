import glob

files = glob.glob('*/*.csv')
print(len(files))
max_height, max_width = 0,0

for fil in files:
    print(fil)
    with open(fil, 'r') as f:
        f.readline()
        for line in f:
            line = line.split(';')
            if max_width < int(line[1]):
                max_width = int(line[1])
            if max_height < int(line[2]):
                max_height = int(line[2])

print(max_width, max_height)

from PIL import Image

img = Image.open('/scratch/jcava/GTSRB/GTSRB/Training/00000/00000_00000.ppm')
img = img.resize((64,64))

import numpy as np
img = np.array(img)
print(img.shape)


###
# Model Checking
###

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3,3))
        self.conv2 = nn.Conv2d(5,5, (3,3))
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(5, 10,(3,3))
        self.conv4 = nn.Conv2d(10, 15, (3,3))
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(15, 15, (3,3))
        self.conv6 = nn.Conv2d(15, 20, (3,3))
        self.pool3 = nn.MaxPool2d(2)
        # self.linear = nn.Linear(320, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = self.pool3(x)

        # x = self.linear(x.flatten())
    
        return x.flatten()

import torchvision.transforms as transforms
transform = transforms.ToTensor()

img = transform(img).unsqueeze(0)

model = Model()
pred = model(img)
print(pred.size())
