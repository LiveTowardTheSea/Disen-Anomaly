# 用于提取和边无关连接无关的特征
import torch
import torch.nn as nn
import math
class Node_Extractor(nn.Module):
    def __init__(self,feature_dim=140, node_channel=2, k_dim=16, dropout=0.35):
        super().__init__()
        self.extractor = nn.Linear(feature_dim, node_channel*k_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(float(self.extractor.weight.size()[1]))
        self.extractor.weight.data.uniform_(-stdv, stdv)
        self.extractor.bias.data.uniform_(-stdv, stdv)    
    
    def forward(self, node_feature):
        # node_feature = torch.cat((node_feature,adj),dim=-1)
        feature = self.dropout(self.extractor(node_feature))   #(n,channel*k_dim)  # 如何确保这些因素之间存在一些独立性
        return feature

