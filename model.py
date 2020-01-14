import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(768, 1000)
        self.fc2 = nn.Linear(1000, 768)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
 
    def gelu(self,x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LinearProject(nn.Module):
    def __init__(self):
        super(LinearProject, self).__init__()
        
        self.fc = nn.Linear(768, 768)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x
