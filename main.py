import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import *
from torchvision import transforms

import torchvision
import torchvision.models as models
import PIL

import sklearn
##
# Argparse
##
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--loss')
parser.add_argument('--param')
parser.add_argument('--attack')
args = parser.parse_args()
##
# Seed
##
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

###
# Model Checking
###

import torch
import torch.nn as nn
import torch.nn.functional as F

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# gtsrb_dataset_train = GTSRB(root_dir='/scratch/jcava/GTSRB/GTSRB/Training')
# loader = torch.utils.data.DataLoader(gtsrb_dataset_train,
#                                              batch_size=len(gtsrb_dataset_train), shuffle=True,
#                                              num_workers=8)
# data = next(iter(loader))
# mean, std = data[0].mean(), data[0].std()
# print(mean,std)

batch_size = 128
gtsrb_dataset_train = GTSRBImbalance(root_dir='/scratch/jcava/GTSRB/GTSRB/Training', minority=14, training=True,
                                            transform=transforms.Compose([transforms.RandomApply([
                                                transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),
                                                transforms.RandomAffine(0, translate=(0.2, 0.2),
                                                                        resample=PIL.Image.BICUBIC),
                                                transforms.RandomAffine(0, shear=20, 
                                                                        resample=PIL.Image.BICUBIC),
                                                transforms.RandomAffine(0, scale=(0.8, 1.2), 
                                                                        resample=PIL.Image.BICUBIC)
                                            ]),
                                            transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(gtsrb_dataset_train,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=8)

gtsrb_dataset_test = GTSRBImbalance(root_dir='/scratch/jcava/GTSRB/GTSRB/Training', minority=14, training=False)

test_dataset = torch.utils.data.DataLoader(gtsrb_dataset_test,
                                             batch_size=1, shuffle=True,
                                             num_workers=8)

###
# Initial Training
###
model = model.cuda()
from losses import *
criterion = None
if args.loss == 'CE':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'ALPHA':
    criterion = AlphaLoss(classes=2, params={'alpha' : float(args.param)})
elif args.loss == 'FOCAL':
    criterion = FocalLoss(params={'gamma' : float(args.param)})

# from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import GradientSignAttack

# adversary = LinfPGDAttack(model, loss_fn=criterion, eps=0.3, nb_iter=10, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary = GradientSignAttack(model, loss_fn=criterion, eps=0.3, clip_min=0.0, clip_max=1.0, targeted=False)


import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-3)
max_epochs = 16
import time
from tqdm import tqdm
print(len(dataset_loader))
loss_iteration = []

for epoch in range(max_epochs):
    start = time.time()
    loss_iteration_base = []
    # Regular Training
    for i, (x,y) in tqdm(enumerate(dataset_loader)):
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        # print(torch.isnan(x).any())
        loss_iteration_base.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    loss_iteration.append(np.mean(loss_iteration_base))
    end = time.time()
    print('Epoch ' + str(epoch) + ': ' + str(end-start) + 's')

for epoch in range(max_epochs):
    start = time.time()
    loss_iteration_adv = []
    # Adversarial Training
    for i, (x,y) in tqdm(enumerate(dataset_loader)):
        x = x.cuda()
        y = y.cuda()
        adv_untargeted = adversary.perturb(x, y)
        pred = model(adv_untargeted)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        # print(torch.isnan(x).any())
        loss_iteration_adv.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    loss_iteration.append(np.mean(loss_iteration_adv))
    end = time.time()
    print('Epoch ' + str(epoch) + ': ' + str(end-start) + 's')

##
# Plot
##
plt.figure()
plt.plot(list(range(len(loss_iteration))), loss_iteration)
plt.savefig('at_resnet18_imbalance_loss-'+str(args.loss)+'-'+str(args.param)+'.png')

###
# AT Testing
###
model.eval()
print(len(test_dataset))
acc = 0
predictions = []
true = []
for i, (x,y) in tqdm(enumerate(test_dataset)):
    x = x.cuda()
    y = y.cuda()
    pred = model(x)
    pred = torch.argmax(pred,dim=1).item()
    # print(pred.item(), y.squeeze(0).item())
    # print(pred)
    predictions.append(pred)
    true.append(y.item())
    if pred == y.item():
        acc += 1
print('Adversarial Training Accuracy: ' + str(float(acc/len(test_dataset))))

from sklearn.metrics import confusion_matrix

f1_score = sklearn.metrics.f1_score(true, predictions, average='binary')
print('Adversarial Training F1 Score: ' + str(f1_score))

import seaborn as sns

plt.figure()

cf_matrix = confusion_matrix(true, predictions)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.5f')

ax.set_title('AT Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Minority','Majority'])
ax.yaxis.set_ticklabels(['Minority','Majoirty'])

## Display the visualization of the Confusion Matrix.
plt.savefig('at-confusion-matrix-'+str(args.loss)+'-'+str(args.param)+'.png')

criterion = nn.CrossEntropyLoss()
import torchattacks

adversary = None
attack = args.attack
if attack == 'FGSM':
    adversary = GradientSignAttack(model, loss_fn=criterion, eps=0.3, clip_min=0.0, clip_max=1.0, targeted=False)
if attack == 'Square':
    adversary = torchattacks.Square(model, norm='Linf', n_queries=50, n_restarts=1, eps=None, p_init=.8, seed=0, verbose=False, targeted=False, loss='margin', resc_schedule=True)
if attack == 'Pixle':
    adversary = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=100, max_iterations=50)
    
adv_acc = 0
predictions = []
true = []
for i, (x,y) in tqdm(enumerate(test_dataset)):
    x = x.cuda()
    y = y.cuda()
    adv_untargeted = adversary.perturb(x, y)
    pred = model(adv_untargeted)
    pred = torch.argmax(pred,dim=1).item()
    # print(pred.item(), y.squeeze(0).item())
    # print(pred)
    predictions.append(pred)
    true.append(y.item())
    if pred == y.item():
        adv_acc += 1
print('Adversarial Attack Accuracy: ' + str(float(adv_acc/len(test_dataset))))

f1_score = sklearn.metrics.f1_score(true, predictions, average='binary')
print('Adversarial Attack F1 Score: ' + str(f1_score))


plt.figure()
cf_matrix = confusion_matrix(true, predictions)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.5f')

ax.set_title('AT Adversarial Attack Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Minority','Majority'])
ax.yaxis.set_ticklabels(['Minority','Majoirty'])

## Display the visualization of the Confusion Matrix.
plt.savefig('at-adversarial-attack-confusion-matrix-'+str(args.loss)+'-'+str(args.param)+'.png')
print('Done')
