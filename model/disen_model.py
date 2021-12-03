import  torch.nn as nn
import torch
from module.encoder import *
from module.decoder import *
class Disen_Model(nn.Module):
    def __init__(self, feature_dim, config):
        super().__init__()
        self.encoder = Encoder(feature_dim, config=config)
        self.decoder = Decoder(feature_dim,node_dim=config.nchannel*config.kdim,
                               edge_dim=config.channel*config.kdim, edge_channel=config.channel)
    
    def forward(self, x, neighbor_id):
        n_feature, ne_feature, mutual_info = self.encoder(x, neighbor_id)
        decode_attribute, decode_adj = self.decoder(n_feature, ne_feature, neighbor_id)
        return decode_attribute, decode_adj, mutual_info
