import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size):
        x= 64
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size,x)
        self.fc2 = nn.Linear(x, 1)
        #Overfit example



    def forward(self, features):
        x = F.sigmoid(self.fc1(features.float() ))
        x = F.sigmoid(self.fc2(x))
        return x
