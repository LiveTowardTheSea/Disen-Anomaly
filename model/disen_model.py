import  torch.nn as nn
import torch
from module.encoder import *
from module.decoder import *
class Disen_Model(nn.Module):
    def __init__(self, feature_dim, config):
        super().__init__()
        self.encoder = Encoder(feature_dim, config=config)
        if config.deck != 0:
            final_channel = max(config.channel - (config.n_layer-1) * config.deck, 1)
            final_dim = final_channel * config.kdim
        else:
            final_dim = config.channel*config.kdim

        self.decoder = Decoder(feature_dim, node_dim=config.nchannel*config.kdim,
                               edge_dim=final_dim, edge_channel=config.channel)
    
    def forward(self, x, neighbor_id):
        ne_feature = self.encoder(x, neighbor_id)
        decode_attribute, decode_adj = self.decoder(ne_feature, neighbor_id)
        return decode_attribute, decode_adj
