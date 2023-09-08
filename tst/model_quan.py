import torch.nn as nn
import torch

import numpy as np
import torch
import torch.nn.functional as F

import torch.quantization

class FeaturesLinear(torch.nn.Module):
    
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
    

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims=[ 241,8 ,   4 ,  89 ,  84 ,   3 , 222   ,25  , 17 ,1718 ,7161 , 608  ,  5   , 5,
 1157 ,   5  ,  6 , 251 ,   5,  51 , 111 ,  51], embed_dim=32, mlp_dims=[32,32], dropout=0.2):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
    
model   = WideAndDeepModel()
checkpoint   = torch.load("/data0/lygao/git/oppo/FL_MPI/model_save/2023-09-06-20_48_11/2023-09-06-20_48_11.pth")
model.load_state_dict(checkpoint)
#model.eval()

#for name, param in model.state_dict().items():
#    print(name, param,param.dtype)

#data = torch.rand(size=(241,8 ,   4 ,  89 ,  84 ,   3 , 222   ,25  , #17 ,1718 ,7161 , 608  ,  5   , 5,
# 1157 ,   5  ,  6 , 251 ,   5,  51 , 111 ,  51), requires_grad=False)

#for name, param in model.state_dict().items():
#    print(name, param,param.dtype)
import copy

model.half()
params_dict = copy.deepcopy(model.state_dict())

for name, param in params_dict.items():
    if param.dtype == torch.float16:
        param_i = (param*127).to(torch.int8)
        param_f = param_i.to(torch.float16)/127.0
        # 增加条件，将 param_f 大于1的值截断为1  Assertion `input_val >= zero && input_val <= one` failed.
        param_f = torch.min(param_f, torch.tensor(1.0, dtype=torch.float16))
        params_dict[name] = param_f
    
    
for name, param in params_dict.items():
    print(name, param,param.dtype)
