# 考虑只生成连接矩阵的情况
import torch.nn as nn
import torch.nn.functional as F
import  torch
import math

def generate_mask(neighbor, mask_id):
    return (neighbor == mask_id).bool()

class Decoder(nn.Module):
    def __init__(self, orig_dim, node_dim, edge_dim, edge_channel):
        # 由聚合的特征，反过来推导属性特征如何办
        #（）
        super().__init__()
        self.catch_zero_d = torch.zeros(1, edge_dim)
        #self.attr_decoder = nn.Linear(node_dim + edge_dim, orig_dim)
        #self.str_decoder = nn.Linear(edge_dim*2, 1)
        self.attr_decoder = nn.Linear(edge_dim, orig_dim)
        self.edge_channel = edge_channel
        self.layer_norm = nn.LayerNorm(edge_dim)
 
    def generate_context_feature(self, agg_feature, nb_id):
        # multi-head 很像
        node_num = agg_feature.shape[0]
        feature_dim = agg_feature.shape[-1]
        self.catch_zero_d = self.catch_zero_d.to(agg_feature.device)
        nb_feature = torch.cat((agg_feature,self.catch_zero_d),dim=0)   # device
        nb_feature = nb_feature[nb_id].view(node_num, -1, feature_dim) 
        attention_score = torch.einsum('nd,nmd->nm',agg_feature, nb_feature)
        attention_mask = generate_mask(nb_id.view(node_num, -1), -1)
        attention_score = torch.masked_fill(attention_score, attention_mask, value=-1e10)
        attention_score = torch.softmax(attention_score/math.sqrt(feature_dim), dim= -1)
        attention_vector = torch.einsum('nm,nmd->nd',attention_score, nb_feature)
        return attention_vector #(n,d)

    def generate_multi_head_feature(self,agg_feature,nb_id):
        node_num = agg_feature.shape[0]
        feature_dim = agg_feature.shape[-1]
        self.catch_zero_d = self.catch_zero_d.to(agg_feature.device)
        nb_feature = torch.cat((agg_feature,self.catch_zero_d),dim=0)   # device
        nb_feature = nb_feature[nb_id].view(node_num, -1, feature_dim)
        agg_feature = agg_feature.view(node_num,self.edge_channel,-1)   
        nb_feature = nb_feature.view(node_num,-1,self.edge_channel,feature_dim//self.edge_channel)
        attention_score = torch.einsum('nhd,nmhd->nmh',agg_feature, nb_feature)
        attention_mask = generate_mask(nb_id.view(node_num, -1), -1)
        attention_score = torch.masked_fill(attention_score, attention_mask.unsqueeze(-1), value=-1e10)
        attention_score = torch.softmax(attention_score/math.sqrt(feature_dim), dim=-2)
        attention_vector = torch.einsum('nmh,nmhd->nhd',attention_score, nb_feature)
        attention_vector = torch.sum(attention_vector,dim=-2)
        return attention_vector #(n,d)


    def generete_attribute(self, agg_feature, nb_id):
        context_feature = self.generate_context_feature(agg_feature, nb_id)
        # # # # # node_feature = torch.cat((node_feature,context_feature),dim=-1)
        node_feature = context_feature
        # node_feature = agg_feature
        # generate_node_feature = torch.sigmoid(self.attr_decoder(node_feature))
        generate_node_feature = F.relu(self.attr_decoder(node_feature))
        return generate_node_feature

    def generate_adj(self,agg_feature):
        # 将 agg_feature 进行归一化
        # agg_feature = self.layer_norm(agg_feature)
        # node_feature = self.layer_norm(node_feature)
        agg_feature_trans = agg_feature.transpose(0,1)
        edge_prob = torch.matmul(agg_feature, agg_feature_trans)
        edge_prob = torch.sigmoid(edge_prob)
        return edge_prob

    def generate_disen_adj(self, agg_feature):
        sample_num = agg_feature.shape[0]
        agg_feature = agg_feature.view(sample_num, self.edge_channel, -1)
        edge_channel_prob = torch.einsum('bcd,scd->bsc',agg_feature, agg_feature) #(b,b,c)
        prob = torch.softmax(edge_channel_prob,dim=-1)
        return prob

    def generate_adj_linear(self, agg_feature):
        # (B,d)
        sample_size = agg_feature.shape[0]
        dim_size = agg_feature.shape[1]
        edge_prob = []
        key_vec = agg_feature
        for i in range(sample_size):
            query_vec = agg_feature[i]
            query_vec = query_vec.unsqueeze(0).expand(sample_size,dim_size)
            qk = torch.cat((query_vec,key_vec),dim=-1)
            prob = self.str_decoder(qk)  #(B,1)
            edge_prob.append(prob)
        edge_prob = torch.cat(edge_prob,dim=-1).transpose(0,1)
        return torch.sigmoid(edge_prob)
        
    def forward(self,agg_feature, nb_id):
        # agg_feature:聚合之后的特征（node, channel*k_dim）
        # 使用 邻居节点重建该节点的属性
        decode_attribute = self.generete_attribute( agg_feature, nb_id)
        decode_adj = self.generate_adj(agg_feature)
        return decode_attribute, decode_adj
