import torch
import torch.nn as nn


class DNN_REC(nn.Module):
    def __init__(self, field_dims):
        super(DNN_REC, self).__init__()
        self.fc1 = nn.Embedding(sum(field_dims), 1)
        self.relu1 = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(512, 128)
        # self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sum(x, dim=1)
        # x = self.relu2(self.fc2(x))
        # x = self.fc3(x)
        x = x.squeeze(dim=1)
        x = torch.sigmoid(x)
        return x