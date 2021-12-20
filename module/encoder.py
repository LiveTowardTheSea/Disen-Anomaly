import  torch 
import torch.nn as nn
from module.gnn import *
from module.mi_reg import *
class Encoder(nn.Module):
    def __init__(self, feature_dim, config):
        super().__init__()
        if config.gnn == 'disen':
            self.ne_extractor = DisenGCN(feature_dim=feature_dim,nlayer=config.n_layer,first_channel=config.channel,
                                        k_dim=config.kdim, dec_k=config.deck, dropout=config.dropout, routit=config.routit, 
                                        tau=config.tau,every_linear=config.every_linear,jump=config.jump)
        else:
            self.ne_extractor = GAT(feature_dim=feature_dim, nlayer=config.n_layer, first_head=config.channel,
                                    head_dim=config.kdim, dropout=config.dropout, every_linear=config.every_linear)
  
    def forward(self, node_feature, neighbour_id):
        ne_feature = self.ne_extractor(node_feature, neighbour_id)
        return ne_feature