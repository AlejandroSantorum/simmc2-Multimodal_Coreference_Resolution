import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        logit = nn.Sigmoid()(input)
        logit = logit.clamp(self.eps, 1. - self.eps)
        
        loss = - target * torch.log(logit) * (1 - logit)**self.gamma * self.alpha- (1 - target) * torch.log(1 - logit) * logit ** self.gamma

        return loss.sum()

if __name__ == '__main__':
    focal = FocalLoss(gamma = 2)
    BCE = torch.nn.BCEWithLogitsLoss()
    A = torch.tensor([4]).float()
    B = torch.tensor([0]).float()
    print(focal(A,B))
    print(BCE(A,B))
