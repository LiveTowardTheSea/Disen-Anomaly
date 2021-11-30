import networkx as nx
import  numpy as np
import pickle

# 可以改进的地方: 对于不满足nb_Size的一些节点，可以不用每次resample的时候都去遍历，可以直接放在init函数里面
class Neigh_Sampler():
    def __init__(self,neighbor_size=30, include_self=False):
        self.neighbor_size = neighbor_size
        self.include_self = include_self


    def get_sample(self, graph,keep_big_degree=False):
        # graph: networkX类型的数据
        # 关于邻居节点的采样问题
        # 返回 np.array [n,nbsz]  编号
        nb_size = self.neighbor_size+1 if self.include_self else self.neighbor_size
        nb_all = np.zeros((graph.number_of_nodes(), nb_size),dtype=np.int64)
        if  self.include_self:
            nb_all[:, 0] = np.arange(nb_all.shape[0], dtype=np.int64)
            nb = nb_all[:, 1:]
        else:
            nb = nb_all
        # nb (n, neighbor_size)
        for u in graph.nodes():
            all_neighbors = sorted(list(graph.neighbors(u)))
            if len(all_neighbors) <= self.neighbor_size:
                neighbors = all_neighbors + ([-1]*(self.neighbor_size-len(all_neighbors)))
            if len(all_neighbors) > self.neighbor_size:
                # 随机采样30个
                # print('当前节点邻居个数为: %d'%len(all_neighbors))
                if keep_big_degree:
                    v_nb = sorted(all_neighbors, key=(lambda x: -graph.degree(x)))
                    random_neigh = np.array(v_nb[:self.neighbor_size], dtype=np.int64)
                else:
                    random_neigh = np.random.choice(all_neighbors, self.neighbor_size, replace=False)
                neighbors = random_neigh
            nb[u] = np.array(neighbors, dtype=np.int64)
        return nb_all
