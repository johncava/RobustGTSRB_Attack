
import torch
import torch.nn.functional as F

# One-hot encoding for 'target' with K classes
def to_one_hot(target, K):
    batch_size = len(target)
    Y = torch.zeros(batch_size, K)
    Y[range(batch_size), target] = 1
    return Y.to(torch.float32)

def alphaloss(output,target, params, device):

    my_alpha = params['alpha']
    loss = 0
    if my_alpha == 1.0:
        loss = torch.mean(torch.sum(-target*torch.log(torch.softmax(output,dim=1) + 1e-8), dim=1))
    else:
        alpha = torch.FloatTensor([my_alpha]).to(device)
        one = torch.FloatTensor([1.0]).to(device)
        loss = (alpha/(alpha-one))*torch.mean(torch.sum(target*(one - torch.softmax(output,dim=1).pow(one - (one/alpha))),dim=1))
    return loss

class AlphaLoss(torch.nn.Module):

    def __init__(self, classes, params):
        super(AlphaLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = classes
        self.params = params

    def forward(self, output, target):

        target_onehot = to_one_hot(target, self.classes)

        return alphaloss(output, target_onehot.to(self.device), self.params, self.device)


class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
        NOTE: Gokul_prasad implementation https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/15 was better
    '''

    def __init__(self, params):
        super(FocalLoss, self).__init__()
        self.gamma = params['gamma']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input, target):
        self.gamma = torch.tensor(self.gamma).float().to(self.device)
        ce_loss = F.cross_entropy(input, target,reduction='mean') 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss