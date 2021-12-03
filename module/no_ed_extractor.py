import  torch.nn as nn
import torch.nn.functional as F
import torch
from module.routing_layer import Routing
import math
 
class SparseInput(nn.Module):
    def __init__(self, input_dim, output_dim,dropout):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(float(self.linear.weight.size()[1]))
        self.linear.weight.data.uniform_(-stdv, stdv)
        self.linear.bias.data.uniform_(-stdv, stdv)    
        # !!! 初始化
    def forward(self, x):
        return self.dropout(F.relu(self.linear(x)))


class DisenGCN(nn.Module):
    def __init__(self, feature_dim=140, nlayer=6, first_channel=8, k_dim=16, 
                 nclass=7, dec_k =1, dropout=0.35, routit=5, tau=1.0, jump=True,every_linear=True):
        super().__init__()
        cur_dim = first_channel * k_dim
        #cur_dim = 0
        self.nlayer = nlayer
        self.jump = jump
        self.pca = SparseInput(feature_dim,first_channel*k_dim, dropout=dropout)
        self.layers = nn.ModuleList()
        in_caps = None
        out_caps = first_channel
        for i in range(self.nlayer):
            self.layers.append(Routing(k_dim, out_caps, in_caps, routit, tau, every_linear))
            cur_dim += out_caps * k_dim
            in_caps = out_caps
            out_caps = max(1, in_caps-dec_k)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, feature, neighbor_id):
        # feature:(n, input_dim)
        x = feature
        neighbor_id = neighbor_id.view(-1)
        xs = []
        x = self.pca(x)
        xs.append(x)
        for  i in range(self.nlayer):
            x = self.dropout(F.relu(self.layers[i](x,neighbor_id)))
            xs.append(x)
        hidden = x
        if self.jump:
            hidden = torch.cat(xs,dim=1)
        #distribution = self.classifier(hidden)
        return hidden
        