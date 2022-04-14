import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import *
from torchvision import transforms

import torchvision
import torchvision.models as models

###
# Model Checking
###

import torch
import torch.nn as nn
import torch.nn.functional as F

import PIL

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 43)

# gtsrb_dataset_train = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training')
# loader = torch.utils.data.DataLoader(gtsrb_dataset_train,
#                                              batch_size=len(gtsrb_dataset_train), shuffle=True,
#                                              num_workers=8)
# data = next(iter(loader))
# mean, std = data[0].mean(), data[0].std()
# print(mean,std)

batch_size = 128
gtsrb_dataset_train = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training', training=True,
                                            transform=transforms.Compose([transforms.RandomApply([
                                                transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),
                                                transforms.RandomAffine(0, translate=(0.2, 0.2),
                                                                        resample=PIL.Image.BICUBIC),
                                                transforms.RandomAffine(0, shear=20, 
                                                                        resample=PIL.Image.BICUBIC),
                                                transforms.RandomAffine(0, scale=(0.8, 1.2), 
                                                                        resample=PIL.Image.BICUBIC)
                                            ]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]))
dataset_loader = torch.utils.data.DataLoader(gtsrb_dataset_train,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=8)

gtsrb_dataset_test = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training', training=False, transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])                                            
)

test_dataset = torch.utils.data.DataLoader(gtsrb_dataset_test,
                                             batch_size=1, shuffle=True,
                                             num_workers=8)

###
# Initial Training
###

model = model.cuda()
criterion = nn.CrossEntropyLoss()
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
max_epochs = 16
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
plt.savefig('baseline_resnet18_loss.png')

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
