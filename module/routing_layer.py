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



        

