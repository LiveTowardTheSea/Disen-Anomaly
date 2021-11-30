import  torch 
import torch.nn as nn
from module.node_extractor import *
from module.no_ed_extractor import *
from module.mi_reg import *
class Encoder(nn.Module):
    def __init__(self, feature_dim, config):
        super().__init__()
        self.n_extractor = Node_Extractor(feature_dim=feature_dim, node_channel=config.nchannel, 
                                                                      k_dim=config.kdim,dropout=config.dropout)
        self.ne_extractor = DisenGCN(feature_dim=feature_dim,nlayer=config.n_layer,first_channel=config.channel,
                                    k_dim=config.kdim, dec_k=config.deck, dropout=config.dropout, routit=config.routit, 
                                    tau=config.tau,every_linear=config.every_linear,jump=config.jump)
        self.mi_reg = Mine(config.kdim, config.mutual_hidden, config.ind_channel, config.mutual_batch)

    def forward(self, node_feature, neighbior_id):
        n_feature = self.n_extractor(node_feature)
        ne_feature = self.ne_extractor(node_feature, neighbior_id)
        mutual_info = self.mi_reg.compute_mutual(n_feature, ne_feature)
        return n_feature, ne_feature, mutual_info