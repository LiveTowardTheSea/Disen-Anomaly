# 论文 mine里的正则化项约束
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class Mine(nn.Module):
    def __init__(self,k_dim, mutual_hidden, ind_channel, mutual_batch):
        super().__init__()
        self.k_dim = k_dim
        self.mutual_hidden = mutual_hidden
        self.ind_channel = ind_channel
        self.mutual_batch = mutual_batch
        self.fc1 = nn.Linear(k_dim*2,mutual_hidden)
        self.fc2 = nn.Linear(mutual_hidden, mutual_hidden)
        self.fc3 = nn.Linear(mutual_hidden,1)
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

    def sample_batch(self, node_data, edge_data, mode='joint'):
        # [batch,ind_channel*kdim*2] 
        ind_channel = self.ind_channel
        k_dim = self.k_dim
        mutual_batch = self.mutual_batch
        node_channel_num = node_data.shape[-1] // k_dim
        edge_channel_num = edge_data.shape[-1] // k_dim
        node_channel_idx = np.random.choice(range(node_channel_num), size=ind_channel).tolist()
        edge_channel_idx = np.random.choice(range(edge_channel_num), size=ind_channel).tolist()  # return list
        node_channel_data = torch.cat([node_data[:, idx * k_dim: (idx+1) * k_dim] for idx in node_channel_idx], dim=-1)
        edge_channel_data = torch.cat([edge_data[:, idx * k_dim: (idx+1) * k_dim] for idx in edge_channel_idx], dim=-1)
        if mode == 'joint':
            node_batch_idx  = np.random.choice(range(node_data.shape[0]),size=mutual_batch,replace=False).tolist()
            edge_batch_idx = node_batch_idx
        elif mode == 'margin':
            node_batch_idx = np.random.choice(range(node_data.shape[0]),size=mutual_batch,replace=False).tolist()
            edge_batch_idx = np.random.choice(range(node_data.shape[0]),size=mutual_batch,replace=False).tolist()
        else:
            return None
        batch_node_data = node_channel_data.view(-1,ind_channel,k_dim)[node_batch_idx]
        batch_edge_data = edge_channel_data.view(-1,ind_channel,k_dim)[edge_batch_idx]  #(batch,ind,k_dim)
        return torch.cat((batch_node_data, batch_edge_data), dim=-1) #(batch,ind,2*k_dim)

    def compute_mutual(self, node_data, edge_data):
        joint = self.sample_batch(node_data, edge_data, mode='joint')
        margin = self.sample_batch(node_data, edge_data, mode='margin')
        le = self.forward(joint) # (batch, ind, 1)
        re = torch.exp(self.forward(margin))
        mi_lb = torch.mean(le, dim=0) - torch.log(torch.mean(re, dim=0))
        return {'le':le,
                're':re,
                'mi_lb':mi_lb}