import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import *
from torchvision import transforms

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
        self.norm1 = nn.BatchNorm2d(5)
        self.pool1 = nn.MaxPool2d(4)
        self.conv3 = nn.Conv2d(5, 10,(3,3))
        self.conv4 = nn.Conv2d(10, 15, (3,3))
        self.norm2 = nn.BatchNorm2d(15)
        self.pool2 = nn.MaxPool2d(4)
        self.conv5 = nn.Conv2d(15, 15, (3,3))
        self.conv6 = nn.Conv2d(15, 20, (3,3))
        self.norm3 = nn.BatchNorm2d(20)
        self.pool3 = nn.MaxPool2d(4)
        self.linear = nn.Linear(80, 43)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = self.norm1(x)
        x = self.pool1(x)

        x = torch.sigmoid(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        x = self.norm2(x)
        x = self.pool2(x)
        
        x = torch.sigmoid(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        x = self.norm3(x)

        x = self.pool3(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x



# gtsrb_dataset_train = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training')
# loader = torch.utils.data.DataLoader(gtsrb_dataset_train,
#                                              batch_size=len(gtsrb_dataset_train), shuffle=True,
#                                              num_workers=8)
# data = next(iter(loader))
# mean, std = data[0].mean(), data[0].std()
# print(mean,std)

batch_size = 128
gtsrb_dataset_train = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training', training=True)
dataset_loader = torch.utils.data.DataLoader(gtsrb_dataset_train,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=8)

gtsrb_dataset_test = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training', training=False)

test_dataset = torch.utils.data.DataLoader(gtsrb_dataset_test,
                                             batch_size=1, shuffle=True,
                                             num_workers=8)

###
# Initial Training
###

model = Model().cuda()
criterion = nn.CrossEntropyLoss()
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-3)
max_epochs = 100
import time
from tqdm import tqdm
print(len(dataset_loader))
loss_iteration = []
for epoch in range(max_epochs):
    start = time.time()
    epoch_loss = []
    for i, (x,y) in tqdm(enumerate(dataset_loader)):
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        # print(torch.isnan(x).any())
        epoch_loss.append(loss.item())
        loss_iteration.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    loss_iteration.append(np.mean(epoch_loss))
    end = time.time()
    print('Epoch ' + str(epoch) + ': ' + str(end-start) + 's')

##
# Plot
##
plt.plot(list(range(len(loss_iteration))), loss_iteration)
plt.savefig('baseline_loss.png')

###
# Testing
###
model.eval()
print(len(test_dataset))
acc = 0
for i, (x,y) in tqdm(enumerate(test_dataset)):
    x = x.cuda()
    y = y.cuda()
    pred = model(x)
    pred = torch.argmax(pred,dim=1).item()
    # print(pred.item(), y.squeeze(0).item())
    if pred == y.item():
        acc += 1
print('Baseline Accuracy: ' + str(float(acc/len(test_dataset))))
print('Done')
