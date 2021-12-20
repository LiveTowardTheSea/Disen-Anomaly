import  torch.nn as nn
import torch.nn.functional as F
import torch
import math

class Routing(nn.Module):
    def __init__(self, k_dim, out_caps, in_caps,rout_it, tau, every_linear=True):
        super().__init__()
        if in_caps is not None and (in_caps != out_caps or every_linear):
            self.fc = nn.Linear(in_caps*k_dim, out_caps*k_dim)
            self.reset_parameters()
        self.channel, self.k_dim = out_caps, k_dim
        self.d = self.channel * self.k_dim 
        self.rout_it = rout_it
        self.tau = tau
        self._catch_zero_d = torch.zeros(1, self.d)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(float(self.fc.weight.size()[1]))
        self.fc.weight.data.uniform_(-stdv, stdv)
        self.fc.bias.data.uniform_(-stdv, stdv)  

    def forward(self, x, neighbor_id):
        # n*m
        dev = x.device
        if self._catch_zero_d.device != dev:
            self._catch_zero_d = self._catch_zero_d.cuda()
        if hasattr(self,'fc'):
            x = F.relu(self.fc(x))
        n = x.size()[0]
        m = neighbor_id.size()[0] // n
        d, channel, k_dim = self.d, self.channel, self.k_dim
        x = F.normalize(x.view(n,channel, k_dim),dim=-1).view(n,d)
        z = torch.cat((x,self._catch_zero_d),dim=0)    #(n+1,d)
        neighbors = z[neighbor_id].view(n, m, d)
        neighbors = neighbors.view(n,m,channel,k_dim)
        u = x.view(n, channel,k_dim) 
        for clus_num in range(self.rout_it):
            p = torch.sum(u.unsqueeze(1)*neighbors,dim=-1) #(n,m,c)
            p = F.softmax(p/self.tau, dim=-1)
            u = torch.sum(p.unsqueeze(-1) * neighbors, dim=1)
            u += x.view(n,channel,k_dim)
            if clus_num < self.rout_it -1:
                u = F.normalize(u, dim=-1)
        return u.view(n,d)

class GAT_layer(nn.Module):
    def __init__(self, in_dim=100, out_head=5, hdim=25, dropout=0.5, every_linear = False):
        super().__init__()
        self.head = out_head
        self.hdim = hdim
        if in_dim is not None and (every_linear or in_dim != out_head * hdim):
            self.fc = nn.Linear(in_dim, out_head * hdim, bias=False)
        self.atten_vec = nn.Parameter(torch.randn(out_head * hdim * 2, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self,feature, nb_id):
        n = feature.size()[0]
        if hasattr(self,'fc'):
            feature = self.fc(feature)
        z = feature #(n, head*hdim)
        # 获取邻居嵌入
        nb_id = nb_id.view(-1)
        nb_mask = (nb_id.view(n, -1) == -1).unsqueeze(-1).unsqueeze(-1)   # (n,m)
        neighbors = z[nb_id] #(n*m,out_head*hdim)   
        nb_num = neighbors.size()[0] // n
        neighbors = neighbors.view(n,nb_num,-1) 
        z = z.unsqueeze(1) #(n,1,head*hdim)
        z = z.expand(-1,nb_num,-1)  #（n,nb_num,head*hdim)
        # 计算注意力
        z = z.view(n, nb_num, self.head, self.hdim)
        neighbors = neighbors.view(n, nb_num, self.head, self.hdim)
        atten_right_opr = torch.cat((z, neighbors),dim=-1)   #(n,m,head,dim*2)
        atten_left_opr = self.atten_vec.data.view(self.head, self.hdim*2, -1) #(head,dim*2,1)
        atten_score = torch.einsum('hds,nmhd->nmhs',atten_left_opr, atten_right_opr)
        atten_score = F.leaky_relu(atten_score, negative_slope=0.2)
        atten_score = torch.masked_fill(atten_score, nb_mask, value=-1e10)
        atten_score = F.softmax(atten_score, dim=1) #(n,m,h,1)
        # 聚合
        aggre_feature = torch.sum(atten_score * neighbors,dim=1) #(n,h,dim)
        aggre_feature = F.elu(aggre_feature.view(n,-1))
        return self.dropout(aggre_feature)
