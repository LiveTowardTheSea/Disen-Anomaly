# 跟模型计算相关的部分：正确率、loss什么的，模型的输入输出计算
from pickle import load
from scipy.sparse import data
from model.disen_model import Disen_Dominant, Disen_DAE
from data.neighbor_sample import Neigh_Sampler
import  torch
import os
import sklearn
import numpy as np
import metric
import torch.nn as nn
import  torch.nn.functional as F

class DisenHelper(object):
    def __init__(self, data_reader, config):
        super().__init__()
        self.data_reader = data_reader
        self.config = config
        self.n, self.d, self.c = data_reader.get_ndc() 
        if config.frame == 'dominant':
            self.model = Disen_Dominant(feature_dim=self.d, config=config)
        elif config.frame == 'dae':
            self.model = Disen_DAE(node_num=self.n, feature_dim=self.d, config=config)
        else:
            assert False, 'no other frame now, please go on coding'
        self.nei_sampler = Neigh_Sampler(neighbor_size=config.nbsz,include_self=config.include_self)
        nb_id = self.nei_sampler.get_sample(self.data_reader.get_graph(), keep_big_degree=True)
        nb_id = torch.from_numpy(nb_id).long()
        feature = self.data_reader.get_feature()
        adj = self.data_reader.get_adj()
        label = self.data_reader.get_label()
        structure_label = self.data_reader.get_structure_label()
        attribute_label = self.data_reader.get_attribute_label()
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(label).long()  #(n,c)
        adj = torch.from_numpy(adj).float()
        if  self.config.use_cuda:   # 使用cuda
            feature = feature.cuda()
            adj = adj.cuda()
            label = label.cuda()
            nb_id = nb_id.cuda()
            if structure_label is not None:
                structure_label = torch.from_numpy(structure_label).long()
                attribute_label = torch.from_numpy(attribute_label).long()
                structure_label = structure_label.cuda()
                attribute_label = attribute_label.cuda()
        self.feature = feature
        self.sample_num = feature.shape[0]
        self.label = label
        self.adj = adj
        self.nb_id = nb_id
        self.structure_label = structure_label
        self.attribute_label = attribute_label
        self.alpha = self.config.alpha
        self.loss_fn = self.att_adj_mutual
        self.mutual_beta = self.config.mutual_beta
        self.get_weight_by_label()

    def initialize_model(self, load_model_path=None):
        # if load_model_path is None:
        #     for param in self.model.parameters():
        #         if param.requires_grad and param.dim()>1:
        #             torch.nn.init.xavier_uniform_(param)
        # else:
        if load_model_path is not None:
            if os.path.isfile(load_model_path):
                self.model.load_state_dict(torch.load(load_model_path))
            else:
                assert False, 'load_model_path is not a file'
        else:
            for param in self.model.parameters():
                if param.requires_grad:
                    if param.dim() > 1:
                        nn.init.xavier_normal_(param)

        if  self.config.use_cuda:
            self.model = self.model.cuda()
        num_param = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name,": ", param.shape)
                num_param += param.numel()
        print('model initialization finished: %d parammeters in total'%num_param)

    def get_weight_by_label(self):
        attr_weight = torch.where(self.feature != 0.0, self.config.ita, 1.0)
        adj_weight = torch.where(self.adj != 0.0, self.config.theta, 1.0 )
        self.attr_weight = attr_weight
        self.adj_weight = adj_weight

    def att_adj_mutual(self):
        att_dis = (self.feature - self.decode_attribute) * self.attr_weight
        att_dis = torch.sqrt(torch.sum(torch.pow(att_dis, 2), 1))
        adj_dis = (self.adj - self.decode_adj) * self.adj_weight
        adj_dis = torch.sqrt(torch.sum(torch.pow(adj_dis, 2), 1))
        # mi_lb = torch.sum(self.mutual_info['mi_lb'])
        loss = self.alpha * att_dis +(1-self.alpha) * adj_dis
        # 求均值
        att_dis = torch.mean(att_dis)
        adj_dis = torch.mean(adj_dis)
        loss = torch.mean(loss)
        return {'loss':loss,
                'att_loss': att_dis,
                'adj_loss': adj_dis}


    # 计算包括训练集在内的所有结果，存入属性中
    # resample  在每一次forward中采样不同的节点
    def forward(self, keep_big_degree=False):
        if self.config.resample and not keep_big_degree:
            neighbor_id = self.nei_sampler.get_sample(self.data_reader.get_graph())
            neighbor_id = torch.from_numpy(neighbor_id).long()
            if self.config.use_cuda:
                neighbor_id = neighbor_id.cuda()
        else:
            neighbor_id = self.nb_id
        decode_attribute, decode_adj = self.model(self.feature, neighbor_id.view(-1)) 
        self.decode_attribute = decode_attribute
        self.decode_adj = decode_adj

    def compute_loss(self):
        # label (n,c) nd_array
        self.model.train()
        # 关于class_weight 的一些东西
        self.forward()
        return self.loss_fn()

    def compute_score(self, print_detail):
        adj_distance = torch.pow((self.decode_adj - self.adj) * self.adj_weight, 2)
        adj_score = torch.sum(adj_distance, dim=-1,keepdim=True)   # (n,1)
        att_distance = torch.pow((self.decode_attribute - self.feature) * self.attr_weight, 2)
        att_score = torch.sum(att_distance, dim=-1,keepdim=True)   # (n,1)  
        score = self.alpha * torch.sqrt(att_score) + (1-self.alpha) * torch.sqrt(adj_score)
        if print_detail:
            save_adj_score = adj_score.cpu().detach().numpy()  #(n,1)
            save_att_score = att_score.cpu().detach().numpy()
            save_score = score.cpu().detach().numpy()
            np.save('plot/adj_score',save_adj_score)
            np.save('plot/att_score',save_att_score)
            np.save('plot/score',save_score)
        return score

    def label_by_rank(self, pred_score):
        # the way of get label by rank
        max_anom_num = self.config.nrank
        pred_label = torch.zeros_like(pred_score).long()  #(n,1)
        _, idx = torch.sort(pred_score, dim=0, descending=True) #(n,1)
        idx = idx.squeeze(-1)
        anomaly_idx = idx[:max_anom_num]
        pred_label[anomaly_idx, 0] = 1
        return anomaly_idx, pred_label # 返回根据最大值生成的idx, 
        
    def get_metric(self, print_detail=False):
        self.model.eval()
        #score = torch.sigmoid(self.logits[idx])
        self.forward(keep_big_degree=True) #(n, n)
        pred_score = self.compute_score(print_detail)  #(n,1)
        anomaly_idx, pred_label = self.label_by_rank(pred_score) #(n,1) gpu
        res = self.compute_matric(pred_score, pred_label, anomaly_idx)
        return res

    def compute_matric(self, pred_score, pred_label, anomaly_idx):
        if self.config.use_cuda:
            pred_label = pred_label.cpu().numpy()    # unknwon whether use detach or not
            targ_label = self.label.cpu().numpy()
            pred_score = pred_score.detach().cpu().numpy()
            anomaly_idx = anomaly_idx.detach().cpu().numpy()
        # 预测对的结构异常的数目
        pred_right_s, pred_right_a = (0,0)
        if self.structure_label is not None:
            targ_s_label = self.structure_label.cpu().numpy()
            targ_a_label = self.attribute_label.cpu().numpy()
            pred_right_s = np.sum((pred_label == 1)*(pred_label == targ_s_label))
            pred_right_a = np.sum((pred_label == 1)*(pred_label == targ_a_label))
        evaluation = metric.eval_pr(pred_score, pred_label, targ_label, anomaly_idx)
        evaluation['structure_num'] = pred_right_s
        evaluation['attribute_num'] = pred_right_a
        return evaluation




        





        

        










