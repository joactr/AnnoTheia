import torch
import torch.nn as nn
import torch.nn.functional as F

class LossAV(nn.Module):
    def __init__(self):
        super(LossAV, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(256, 2)

    def forward(self, x, labels=None):
        x = x.squeeze(1)
        x = self.fc(x)
        if labels == None:
            predScore = F.softmax(x, dim = -1)
            predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
            return predScore, predLabel
        else:
            nloss = self.criterion(x, labels)
            predScore = F.softmax(x, dim = -1)
            predLabel = torch.round(F.softmax(x, dim = -1))[:,1]

            correctNum = (predLabel == labels).sum().float()
            return nloss, predScore, predLabel, correctNum

class LossA(nn.Module):
    def __init__(self):
        super(LossA, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(128, 2)

    def forward(self, x, labels):
        x = x.squeeze(1)
        x = self.fc(x)
        nloss = self.criterion(x, labels)
        return nloss

class LossV(nn.Module):
    def __init__(self):
        super(LossV, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(128, 2)

    def forward(self, x, labels):
        x = x.squeeze(1)
        x = self.fc(x)
        nloss = self.criterion(x, labels)
        return nloss
