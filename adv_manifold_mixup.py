##
# Adversarial Manifold Mixup Implementation
##

from dataset import *

###
# Model Checking
###

import torch
import torch.nn as nn
import torch.nn.functional as F

from advertorch.attacks import GradientSignAttack

from losses import AlphaLoss

class Checkpoint1(nn.Module):
    def __init__(self, conv3, conv4, conv5, conv6, linear):
        super(Checkpoint1, self).__init__()
        self.conv3 = conv3
        self.conv4 = conv4
        self.pool2 = nn.MaxPool2d(4)
        self.conv5 = conv5
        self.conv6 = conv6
        self.pool3 = nn.MaxPool2d(4)
        self.linear = linear

    def forward(self, x):
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = self.pool3(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

class Checkpoint2(nn.Module):
    def __init__(self, conv5, conv6, linear):
        super(Checkpoint2, self).__init__()
        self.conv5 = conv5
        self.conv6 = conv6
        self.pool3 = nn.MaxPool2d(4)
        self.linear = linear

    def forward(self, x):      
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = self.pool3(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

class Checkpoint3(nn.Module):
    def __init__(self, linear):
        super(Checkpoint3, self).__init__()
        self.linear = linear

    def forward(self, x):      
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3,3))
        self.conv2 = nn.Conv2d(5,5, (3,3))
        self.pool1 = nn.MaxPool2d(4)
        self.conv3 = nn.Conv2d(5, 10,(3,3))
        self.conv4 = nn.Conv2d(10, 15, (3,3))
        self.pool2 = nn.MaxPool2d(4)
        self.conv5 = nn.Conv2d(15, 15, (3,3))
        self.conv6 = nn.Conv2d(15, 20, (3,3))
        self.pool3 = nn.MaxPool2d(4)
        self.linear = nn.Linear(80, 43)

        self.loss_fn = AlphaLoss(classes=43, params={'alpha' : 1.0})

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
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

    def checkpoint1(self,x):
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = self.pool3(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

    def checkpoint2(self,x):
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = self.pool3(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

    def checkpoint3(self,x):
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

    def forward_attack(self,x,y,check):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool1(x)

        if check == 0:
            print('Checkpoint1')
            check1 = Checkpoint1(self.conv3, self.conv4, self.conv5, self.conv6, self.linear)
            adversary = GradientSignAttack(check1, loss_fn=self.loss_fn, eps=0.3, clip_min=0.0, clip_max=1.0, targeted=False)
            adv_untargeted = adversary.perturb(x, y)
            alpha = torch.rand(1).half().cuda()
            mix = alpha*x + (torch.ones(1).half().cuda() - alpha)*adv_untargeted
            x = self.checkpoint1(mix)
        elif check == 1:
            print('Checkpoint2')
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))

            x = self.pool2(x)
            check2 = Checkpoint2(self.conv5, self.conv6, self.linear)
            adversary = GradientSignAttack(check2, loss_fn=self.loss_fn, eps=0.3, clip_min=0.0, clip_max=1.0, targeted=False)
            adv_untargeted = adversary.perturb(x, y)
            alpha = torch.rand(1).half().cuda()
            mix = alpha*x + (torch.ones(1).half().cuda() - alpha)*adv_untargeted
            x = self.checkpoint2(mix)
        else:
            print('Checkpoint3')
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))

            x = self.pool2(x)
            
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            
            x = self.pool3(x)

            check3 = Checkpoint3(self.linear)
            adversary = GradientSignAttack(check3, loss_fn=self.loss_fn, eps=0.3, clip_min=0.0, clip_max=1.0, targeted=False)
            adv_untargeted = adversary.perturb(x, y)
            alpha = torch.rand(1).half().cuda()
            mix = alpha*x + (torch.ones(1).half().cuda() - alpha)*adv_untargeted
            x = self.checkpoint3(mix)

        return x                

gtsrb_dataset_train = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training')
gtsrb_dataset_test = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training', training=False)

batch_size = 128
dataset_loader = torch.utils.data.DataLoader(gtsrb_dataset_train,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=8)

test_dataset = torch.utils.data.DataLoader(gtsrb_dataset_test,
                                             batch_size=1, shuffle=True,
                                             num_workers=8)
###
# Initial Training
###

model = Model().half().cuda()
criterion = nn.CrossEntropyLoss()
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-2)
max_epochs = 100
import time
from tqdm import tqdm
# print(len(dataset_loader))
# for epoch in range(max_epochs):
#     start = time.time()
#     for i, (x,y) in tqdm(enumerate(dataset_loader)):
#         x = x.half().cuda()
#         y = y.cuda()
#         pred = model(x)
#         optimizer.zero_grad()
#         loss = criterion(pred, y)
#         loss.backward()
#         optimizer.step()
#         break
#     end = time.time()
#     print('Epoch ' + str(epoch) + ': ' + str(end-start) + 's')


###
# Adversarial Training
###
import random
print(len(dataset_loader))
for epoch in range(max_epochs):
    start = time.time()
    for i, (x,y) in tqdm(enumerate(dataset_loader)):
        x = x.half().cuda()
        y = y.cuda()
        c = random.randint(0,2)
        pred = model.forward_attack(x,y,c)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    end = time.time()
    print('Epoch ' + str(epoch) + ': ' + str(end-start) + 's')

print('Done')

###
# Testing
###
print(len(test_dataset))
acc = 0
for i, (x,y) in tqdm(enumerate(test_dataset)):
    x = x.half().cuda()
    y = y.cuda()
    pred = model(x)
    pred = torch.argmax(pred.squeeze(0)).item()
    # print(pred.item(), y.squeeze(0).item())
    if pred == y.squeeze(0).item():
        acc += 1
print('Adversarial Manifold Mixup Training Accuracy: ' + str(float(acc/len(test_dataset))))
print('Done')