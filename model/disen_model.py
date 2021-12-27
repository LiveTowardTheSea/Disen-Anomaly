import  torch.nn as nn
import torch
from module.encoder import *
from module.decoder import *
class Disen_Dominant(nn.Module):
    def __init__(self, feature_dim, config):
        super().__init__()
        self.encoder = Encoder(feature_dim, config=config)
        if config.deck != 0:
            final_channel = max(config.channel - (config.n_layer-1) * config.deck, 1)
            final_dim = final_channel * config.kdim
        else:
            final_dim = config.channel*config.kdim

        self.decoder = Decoder(feature_dim, edge_dim=final_dim, edge_channel=config.channel)
    
    def forward(self, x, neighbor_id):
        ne_feature = self.encoder(x, neighbor_id)
        decode_attribute, decode_adj = self.decoder(ne_feature, neighbor_id)
        return decode_attribute, decode_adj

class Disen_DAE(nn.Module):
    def __init__(self,node_num, feature_dim, config):
        super().__init__()
        self.encoder = DAE_Encoder(node_num, feature_dim, config)
        self.decoder = DAE_Decoder(config.dropout)
    
    def forward(self, x, neighbor_id):
        n_feature, ne_feature = self.encoder(x, neighbor_id)
        decode_attribute, decode_adj = self.decoder(n_feature, ne_feature)
        return decode_attribute, decode_adj

        