import pickle
import os
import sys
from networkx.classes import graph
import scipy.io as sio
import scipy.sparse as scpsps
import numpy as np
import networkx as nx
import urllib.request
from data.inject_anomaly import *

class DataLoader:
    def __init__(self, config):
        self.load_dir = config.raw_data_dir # 加载文件的路径
        self.save_dir = config.save_data_dir # 保存文件的路径
        self.data_name = config.data_name
        self.seed = config.dataseed
        self.m = config.m
        self.num = config.num
        self.k = config.k
        self.feature = None # ndarray  （n,d）
        self.label = None #ndarray           (n,1)
        self.graph = None  # 导入networkX类型的数据
        self.adj = None
        self.structure_label = None
        self.attribute_label = None
        self.orig_label = None   #()
        self.n, self.d, self.c = 0, 0, 0
        self.initialize_member()

    def initialize_member(self):
        # 保存文件到anom_data里面 
        # generate_anomaly(self.load_dir, self.save_dir,self.data_name,self.seed, self.m, self.num, self.k)
        # load_mat_path = os.path.join(self.save_dir, self.data_name, 'data.mat')
        load_mat_path = os.path.join(self.save_dir, self.data_name, 'data_dominant.mat')
        if os.path.isfile(load_mat_path):
            mat = sio.loadmat(load_mat_path)
        else:
            print('file dose not exist')
            exit(0)
        
        feature =  mat['Attributes'] #scipy.csr_matrix
        label = mat['Label']
        adj = mat['Network']             #scipy.csr_matrix
        orig_label = mat['Class']
        # 图
        self.adj = np.array(adj.todense(), dtype=np.float32)
        self.graph = self.adj2nx(adj)
        self.feature = feature.toarray().astype(np.float32)
        
        self.label = label.astype(np.int64) 
        structure_label_num = 0
        attribute_label_num = 0
        self.orig_label = orig_label.astype(np.int64)

        self.n = self.feature.shape[0]
        self.d = self.feature.shape[1]
        self.c = 2
        # 判断是否还有structure 以及Attribute 的标签
        if 'str_anomaly_label' in mat.keys() and 'attr_anomaly_label' in mat.keys():
            structure_label = mat['str_anomaly_label']
            attribute_label = mat['attr_anomaly_label']
            structure_label_num = np.sum(structure_label)
            attribute_label_num = np.sum(attribute_label)
            self.structure_label = structure_label.astype(np.int64)
            self.attribute_label = attribute_label.astype(np.int64)

        print('%d instance, feature_dim:%d, edge:%d,'
             ' class:%d, strucute anomaly:%d, attribute_anomaly:%d'%(self.n, self.d, self.graph.number_of_edges(),
                                                                     self.c, structure_label_num,attribute_label_num))
    def adj2nx(self, adj):
        # 获得关于图的表示
        graph = nx.Graph()
        graph.add_nodes_from(range(adj.shape[0]))
        adj_coo = adj.tocoo()
        for u,v,_ in zip(adj_coo.row, adj_coo.col, adj_coo.data):
            graph.add_edge(u, v)
        assert graph.number_of_nodes() == adj.shape[0]
        return graph

    def get_graph(self):
        return self.graph
    
    def get_feature(self):
        return self.feature

    def get_label(self):
        return self.label

    def get_ndc(self):
        return self.n, self.d, self.c
    
    def get_adj(self):
        return self.adj

    def get_structure_label(self):
        return self.structure_label
    
    def get_attribute_label(self):
        return self.attribute_label
        