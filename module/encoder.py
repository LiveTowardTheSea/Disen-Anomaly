import  torch 
import torch.nn as nn
from module.gnn import *
from module.mi_reg import *
import math
class Encoder(nn.Module):
    def __init__(self,feature_dim, config):
        super().__init__()
        self.gnn = config.gnn
        if config.gnn == 'disen':
            self.ne_extractor = DisenGCN(feature_dim=feature_dim,nlayer=config.n_layer,first_channel=config.channel,
                                        k_dim=config.kdim, dec_k=config.deck, dropout=config.dropout, routit=config.routit, 
                                        tau=config.tau,every_linear=config.every_linear,jump=config.jump)
        elif config.gnn == 'gat':
            self.ne_extractor = GAT(feature_dim=feature_dim, nlayer=config.n_layer, first_head=config.channel,
                                    head_dim=config.kdim, dropout=config.dropout, every_linear=config.every_linear)
        elif config.gnn == 'linear':
            self.ne_extractor = nn.Linear(feature_dim, config.channel*config.kdim)
            self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(float(self.ne_extractor.weight.size()[1]))
        self.ne_extractor.weight.data.uniform_(-stdv, stdv)
        self.ne_extractor.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_feature, neighbour_id):
        if self.gnn != 'linear':
            ne_feature = self.ne_extractor(node_feature, neighbour_id)
        else:
            ne_feature = self.ne_extractor(node_feature)
        return ne_feature


class DAE_Encoder(nn.Module):
    def __init__(self,node_num, feature_dim, config):
        super().__init__()
        self.node_fc1 = nn.Linear(node_num, config.n_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        if config.gnn == 'disen':
            self.ne_extractor = DisenGCN(feature_dim=feature_dim,nlayer=config.n_layer,first_channel=config.channel,
                                k_dim=config.kdim, dec_k=config.deck, dropout=config.dropout, routit=config.routit, 
                                tau=config.tau,every_linear=config.every_linear,jump=config.jump)
        else:
             self.ne_extractor = GAT(feature_dim=feature_dim, nlayer=config.n_layer, first_head=config.channel,
                                     head_dim=config.kdim, dropout=config.dropout, every_linear=config.every_linear)
        
        if config.deck != 0:
            final_channel = max(config.channel - (config.n_layer-1) * config.deck, 1)
            final_dim = final_channel * config.kdim
        else:
            final_dim = config.channel * config.kdim

        self.node_fc2 = nn.Linear(config.n_hidden, final_dim)
        self.dropout2 = nn.Dropout(config.dropout)
        self.reset_parameter()


    def reset_parameter(self):
        stdv = 1. / math.sqrt(float(self.node_fc1.weight.size()[1]))
        self.node_fc1.weight.data.uniform_(-stdv, stdv)
        self.node_fc1.bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(float(self.node_fc2.weight.size()[1]))
        self.node_fc2.weight.data.uniform_(-stdv, stdv)
        self.node_fc2.bias.data.uniform_(-stdv, stdv)


    def forward(self,node_feature, neighbor_id):
        n_feature = node_feature.transpose(0, 1)  #(d,n)
        n_feature = self.dropout1(F.relu(self.node_fc1(n_feature)))
        n_feature = self.dropout2(self.node_fc2(n_feature))
        ne_feature = self.ne_extractor(node_feature, neighbor_id)
        return n_feature, ne_feature


        
