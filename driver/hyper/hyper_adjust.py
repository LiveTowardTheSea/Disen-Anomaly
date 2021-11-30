import sys

from torch._C import TensorType
sys.path.append('/home/wqt/code/DisenGCN/DisenGCN-multi')
import train
from config import Configurable
import numpy as np
import json

def test_deck():
# POS
# train_ratio = 0.8

# [Network]
# n_layer = 2
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45 
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 31

# [Optimizer]
# lr = 0.0057
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 400
# early_stop = 1000
# deck: 0 END in 98.15s--epoch:382---val-micro: 44.64% val-macro:8.58% test-micro: 42.01% test-macro:9.84%
# deck: 1 END in 98.91s--epoch:7---val-micro: 42.90% val-macro:5.58% test-micro: 41.47% test-macro:7.98%
# deck: 2 END in 98.02s--epoch:397---val-micro: 43.53% val-macro:6.88% test-micro: 42.31% test-macro:10.33%
# deck: 3 END in 99.77s--epoch:397---val-micro: 43.69% val-macro:6.83% test-micro: 42.16% test-macro:10.18%
# deck: 4 END in 99.71s--epoch:39---val-micro: 43.06% val-macro:5.95% test-micro: 41.40% test-macro:9.24%
# deck: 5 END in 99.74s--epoch:399---val-micro: 43.06% val-macro:5.96% test-micro: 41.86% test-macro:9.61%
# deck: 6 END in 99.06s--epoch:375---val-micro: 43.06% val-macro:6.00% test-micro: 41.70% test-macro:9.44%
# deck: 7 END in 99.10s--epoch:394---val-micro: 42.59% val-macro:5.25% test-micro: 41.86% test-macro:9.71%
# deck: 8 END in 98.22s--epoch:283---val-micro: 43.53% val-macro:6.90% test-micro: 41.70% test-macro:9.60%
#PPI
# [network]
# n_layer = 2
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0057
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 1000
# early_stop = 500
# deck: 0 END in 231.14s--epoch:260---val-micro: 24.98% val-macro:22.67% test-micro: 23.75% test-macro:19.63%
# deck: 1 END in 292.56s--epoch:475---val-micro: 24.86% val-macro:21.59% test-micro: 23.50% test-macro:20.59%
# deck: 2 END in 300.98s--epoch:498---val-micro: 25.30% val-macro:21.75% test-micro: 22.94% test-macro:19.57%
# deck: 3 END in 169.50s--epoch:334---val-micro: 25.43% val-macro:21.98% test-micro: 23.25% test-macro:20.18%
# deck: 4 END in 163.36s--epoch:297---val-micro: 25.11% val-macro:21.72% test-micro: 23.57% test-macro:20.06%
# deck: 5 END in 201.48s--epoch:663---val-micro: 25.68% val-macro:21.79% test-micro: 22.07% test-macro:18.19%
    deck_list = [0,1,2,3,4,5]
    extra_args = ['--deck','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for deck in deck_list:
        extra_args[-1] = deck
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('deck: %d'%deck,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = deck

    print('------the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%% tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                ,best_test_f1[0]*100, best_test_f1[1]*100))


def test_rout_it():
# train_ratio = 0.8

# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0093
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00036
# clip = 2.0

# [Run]
# epoch = 3000
# early_stop = 1000
# rout: 1 END in 2282.22s--epoch:2279---val-micro: 41.56% val-macro:33.22% test-micro: 39.94% test-macro:28.23%
# rout: 2 END in 1850.12s--epoch:1689---val-micro: 41.20% val-macro:32.75% test-micro: 40.09% test-macro:29.33%
# rout: 3 END in 1463.93s--epoch:1580---val-micro: 41.56% val-macro:31.81% test-micro: 40.66% test-macro:29.60%
# rout: 4 END in 1824.38s--epoch:2084---val-micro: 41.85% val-macro:32.75% test-micro: 40.73% test-macro:28.90%
# rout: 5 END in 1937.39s--epoch:2903---val-micro: 42.14% val-macro:33.57% test-micro: 40.16% test-macro:29.73%
# rout: 6 END in 1584.83s--epoch:1306---val-micro: 42.06% val-macro:32.60% test-micro: 39.66% test-macro:27.63%
# rout: 7 END in 2013.94s--epoch:1782---val-micro: 41.92% val-macro:33.12% test-micro: 38.80% test-macro:28.75%
# rout: 8 END in 2290.53s--epoch:2390---val-micro: 40.98% val-macro:33.68% test-micro: 40.16% test-macro:32.30%
# ------the best is: 5, val-micro: 42.14%, val-macro: 33.57%tst-micro:40.16% tst-macro:29.73%
    rout_list = [1,2,3,4,5,6,7,8]
    extra_args = ['--routit','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for rout in rout_list:
        extra_args[-1] = rout
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('rout: %d'%rout,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = rout

    print('------the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%% tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                ,best_test_f1[0]*100, best_test_f1[1]*100))


def test_layer():
#PPI
# train_ratio = 0.8
# 采用随机种子666，2层是最好的
# [Network]
# channel = 10
# kdim = 12
# deck = 4
# dropout = 0.45 
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1

# [Optimizer]
# lr = 0.0057
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 400
# early_stop = 1000

    # layer1: END in 72.34s--epoch:258/400---val-micro: 26.64% val-macro:22.07% test-micro: 25.24% test-macro:21.73%
    #layer 2: END in 92.88s--epoch:397    --  -val-micro: 31.08% val-macro:26.24% test-micro: 27.48% test-macro:21.86%
    # layer3: END in 115.73s--epoch:278---val-micro: 25.52% val-macro:23.71% test-micro: 24.30% test-macro:22.81%
    #            4:END in 144.85s--epoch:353---val-micro: 26.28% val-macro:20.06% test-micro: 23.85% test-macro:19.03%
    #            5:END in 209.51s--epoch:179---val-micro: 25.88% val-macro:20.00% test-micro: 24.15% test-macro:19.62%
    #            6:END in 220.44s--epoch:273---val-micro: 28.48% val-macro:24.63% test-micro: 24.75% test-macro:20.34%
    #            7:END in 210.05s--epoch:233---val-micro: 27.44% val-macro:21.40% test-micro: 28.07% test-macro:21.62%
    #            8:END in 283.22s--epoch:310---val-micro: 25.49% val-macro:20.63% test-micro: 22.03% test-macro:19.82%
    # 设置了种子之后2层不如7层  （21）
    # 1        END in 67.38s--epoch:394---val-micro: 27.30% val-macro:22.67% test-micro: 25.29% test-macro:20.16%
    # 2：END in 87.32s--epoch:386---val-micro: 27.99% val-macro:23.13% test-micro: 24.46% test-macro:19.43%
    # 3：END in 104.51s--epoch:345---val-micro: 27.47% val-macro:23.01% test-micro: 24.12% test-macro:18.04%
    #4： END in 165.52s--epoch:391---val-micro: 27.82% val-macro:23.31% test-micro: 24.29% test-macro:19.02%
    # 5：END in 162.78s--epoch:315---val-micro: 28.67% val-macro:23.87% test-micro: 23.62% test-macro:18.78%   # the best
    #6：END in 216.45s--epoch:389---val-micro: 27.82% val-macro:23.87% test-micro: 23.62% test-macro:18.31%
    #7： END in 229.04s--epoch:344---val-micro: 27.65% val-macro:23.61% test-micro: 26.30% test-macro:20.71%
    # 8        END in 167.38s--epoch:390---val-micro: 27.30% val-macro:22.39% test-micro: 25.13% test-macro:19.64%
    # 采用一千轮进行训练：
#layer: 1         END in 171.69s--epoch:731---val-micro: 28.84% val-macro:24.57% test-micro: 24.96% test-macro:19.39%  lower
# layer: 2        END in 211.80s--epoch:552---val-micro: 28.50% val-macro:24.15% test-micro: 24.29% test-macro:19.09%   lower
# layer: 3        END in 251.68s--epoch:871---val-micro: 28.33% val-macro:23.78% test-micro: 24.62% test-macro:18.74%    higher
#layer: 4        END in 286.94s--epoch:510---val-micro: 28.50% val-macro:23.62% test-micro: 25.63% test-macro:19.28%     higher
#layer: 5        END in 322.39s--epoch:602---val-micro: 29.52% val-macro:25.04% test-micro: 23.95% test-macro:19.10%     higher
#layer: 6        END in 360.94s--epoch:656---val-micro: 29.52% val-macro:25.39% test-micro: 24.46% test-macro:17.83%      one h one l
# layer: 7        END in 394.15s--epoch:498---val-micro: 27.65% val-macro:23.78% test-micro: 24.62% test-macro:20.00%      lower
# layer: 8        END in 416.14s--epoch:825---val-micro: 27.65% val-macro:23.23% test-micro: 24.29% test-macro:19.26%      lower

# blogcatalog

# [Network]
# n_layer = 2
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45 
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 31

# [Optimizer]
# lr = 0.051
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 1500
# early_stop = 1000

#   layer2:       END in 1439.49s--epoch:1280---val-micro: 41.49% val-macro:29.91% test-micro: 39.27% test-macro:28.15% 
# layer: 4        END in 1787.18s--epoch:1302---val-micro: 42.51% val-macro:31.68% test-micro: 39.69% test-macro:28.09%
# layer: 6        END in 2234.59s--epoch:1350---val-micro: 40.91% val-macro:31.32% test-micro: 39.76% test-macro:29.88%
# 四层够了
    layer_list = [2,4,6,8]
    extra_args = ['--n_layer','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for layer in layer_list:
        extra_args[-1] = layer
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('layer: %d'%layer,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = layer

    print('------the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%%, tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                                                                     ,best_test_f1[0]*100, best_test_f1[1]*100))

def test_seed():
# PPI
# train_ratio = 0.8

# [Network]
# n_layer = 2
# channel = 10
# kdim = 12
# deck = 4
# dropout = 0.45 
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1  

# [Optimizer]
# lr = 0.0057
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 400
# early_stop = 1000       # 数据集的划分也因此发生了一些变化，是不是要确保是同样的划分呢
# seed: 27        END in 84.18s--epoch:196---val-micro: 24.87% val-macro:20.92% test-micro: 22.45% test-macro:18.42%
# seed: 31        END in 85.18s--epoch:304---val-micro: 26.51% val-macro:24.06% test-micro: 28.53% test-macro:24.00%
# seed: 45        END in 87.24s--epoch:249---val-micro: 26.98% val-macro:22.51% test-micro: 25.96% test-macro:22.62%
# seed: 42        END in 87.03s--epoch:238---val-micro: 28.43% val-macro:25.09% test-micro: 25.51% test-macro:20.96%
# seed: 12        END in 85.18s--epoch:196---val-micro: 28.65% val-macro:22.35% test-micro: 26.85% test-macro:22.65%
# seed: 20        END in 83.95s--epoch:241---val-micro: 26.83% val-macro:22.12% test-micro: 28.07% test-macro:23.53%
# seed: 1          END in 83.88s--epoch:313---val-micro: 28.52% val-macro:22.88% test-micro: 25.93% test-macro:23.04%
# seed: 3          END in 86.39s--epoch:397---val-micro: 28.50% val-macro:25.31% test-micro: 22.95% test-macro:18.52%
# seed: 29        END in 84.90s--epoch:398---val-micro: 27.03% val-macro:23.36% test-micro: 24.46% test-macro:21.73%
# seed: 23        END in 85.28s--epoch:384---val-micro: 27.00% val-macro:22.19% test-micro: 26.58% test-macro:22.89%
# seed: 33        END in 86.32s--epoch:289---val-micro: 27.43% val-macro:23.45% test-micro: 25.53% test-macro:21.70%
# seed: 40        END in 86.44s--epoch:203---val-micro: 30.78% val-macro:27.45% test-micro: 23.82% test-macro:22.84%
# seed: 32        END in 86.98s--epoch:374---val-micro: 27.63% val-macro:23.13% test-micro: 23.13% test-macro:19.50%
# seed: 43        END in 86.79s--epoch:396---val-micro: 28.02% val-macro:22.71% test-micro: 26.03% test-macro:23.01%
# seed: 13        END in 86.63s--epoch:354---val-micro: 26.18% val-macro:18.83% test-micro: 25.47% test-macro:19.82%
# seed: 36        END in 86.56s--epoch:336---val-micro: 27.84% val-macro:23.99% test-micro: 22.39% test-macro:17.23%
# seed: 4          END in 86.12s--epoch:321---val-micro: 28.64% val-macro:23.25% test-micro: 24.12% test-macro:19.88%
# seed: 0           END in 84.58s--epoch:214---val-micro: 24.01% val-macro:19.56% test-micro: 27.30% test-macro:19.84%
# seed: 49        END in 84.18s--epoch:391---val-micro: 29.25% val-macro:24.68% test-micro: 22.53% test-macro:18.13%
# seed: 11        END in 86.04s--epoch:383---val-micro: 24.57% val-macro:21.15% test-micro: 27.98% test-macro:22.80%
# seed: 10        END in 84.36s--epoch:306---val-micro: 29.38% val-macro:23.27% test-micro: 26.94% test-macro:19.95%
# seed: 39        END in 83.88s--epoch:357---val-micro: 28.96% val-macro:23.37% test-micro: 25.46% test-macro:20.30%
# seed: 24        END in 84.85s--epoch:336---val-micro: 24.49% val-macro:20.42% test-micro: 27.84% test-macro:22.60%
# seed: 21        END in 86.84s--epoch:386---val-micro: 27.99% val-macro:23.13% test-micro: 24.46% test-macro:19.43%
# seed: 7          END in 85.50s--epoch:385---val-micro: 28.81% val-macro:26.51% test-micro: 24.87% test-macro:21.41%
# seed: 30        END in 84.08s--epoch:368---val-micro: 27.01% val-macro:20.07% test-micro: 24.98% test-macro:21.37%
# seed: 34        END in 85.31s--epoch:276---val-micro: 26.34% val-macro:21.57% test-micro: 27.27% test-macro:23.31%
# seed: 17        END in 84.25s--epoch:315---val-micro: 27.45% val-macro:23.48% test-micro: 20.95% test-macro:17.92%
# seed: 28        END in 85.99s--epoch:399---val-micro: 24.91% val-macro:18.90% test-micro: 25.65% test-macro:24.07%
# seed: 48        END in 85.07s--epoch:278---val-micro: 26.89% val-macro:22.51% test-micro: 27.05% test-macro:23.91%
# seed: 18        END in 84.59s--epoch:366---val-micro: 28.20% val-macro:23.08% test-micro: 25.93% test-macro:21.26%
# seed: 26        END in 86.38s--epoch:374---val-micro: 29.12% val-macro:24.39% test-micro: 26.94% test-macro:21.93%
# seed: 37        END in 85.03s--epoch:399---val-micro: 26.77% val-macro:23.90% test-micro: 24.78% test-macro:19.53%
# seed: 41        END in 86.92s--epoch:379---val-micro: 28.15% val-macro:22.86% test-micro: 27.69% test-macro:23.73%
# seed: 14        END in 87.06s--epoch:387---val-micro: 30.25% val-macro:25.88% test-micro: 24.92% test-macro:21.34%
# seed: 2 END in 84.57s--epoch:191---val-micro: 25.93% val-macro:19.58% test-micro: 25.33% test-macro:21.81%
# seed: 44        END in 85.33s--epoch:286---val-micro: 25.84% val-macro:21.69% test-micro: 26.94% test-macro:21.52%
# seed: 5 END in 84.69s--epoch:273---val-micro: 30.93% val-macro:25.69% test-micro: 22.04% test-macro:16.93%
# seed: 6 END in 85.44s--epoch:344---val-micro: 27.21% val-macro:23.33% test-micro: 24.00% test-macro:20.04%
    seed_list = np.random.permutation(50).tolist()
    extra_args = ['--seed','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for seed in seed_list:
        extra_args[-1] = seed
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('seed: %d'%seed,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = seed

    print('------the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%%, tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                                                                     ,best_test_f1[0]*100, best_test_f1[1]*100))

def test_c_h():
    #POS
# 采用三层，不dec的方式来查看模型到底适合几个频道
#------the best is: 10, val-micro: 43.94%, val-macro: 9.35%tst-micro:45.37%, tst-macro:11.31%     
#PPI
# [Network]
# n_layer = 2
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0057
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 1000
# early_stop = 500
# channel: 5      END in 173.12s--epoch:329---val-micro: 24.92% val-macro:22.03% test-micro: 24.13% test-macro:19.77%
# channel: 6      END in 205.45s--epoch:599---val-micro: 25.87% val-macro:21.79% test-micro: 24.19% test-macro:20.50%
# channel: 7      END in 194.40s--epoch:441---val-micro: 24.92% val-macro:21.61% test-micro: 23.32% test-macro:19.75%
# channel: 8      END in 184.16s--epoch:397---val-micro: 24.79% val-macro:21.58% test-micro: 23.32% test-macro:19.67%
# channel: 9      END in 166.03s--epoch:312---val-micro: 25.36% val-macro:22.42% test-micro: 23.75% test-macro:20.02%
# channel: 10     END in 155.84s--epoch:260---val-micro: 24.98% val-macro:22.67% test-micro: 23.75% test-macro:19.63%
# channel: 11     END in 138.65s--epoch:177---val-micro: 24.61% val-macro:21.05% test-micro: 25.00% test-macro:20.40%
# channel: 12     END in 169.72s--epoch:317---val-micro: 25.24% val-macro:22.11% test-micro: 23.32% test-macro:20.25%
# ------the best is: 9, val-micro: 25.36%, val-macro: 22.42%tst-micro:23.75%, tst-macro:20.02%
# blogcatalog
# train_ratio = 0.8

# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0093
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00036
# clip = 2.0

# [Run]
# epoch = 3000
# early_stop = 1000
# channel: 4      END in 2140.73s--epoch:2633---val-micro: 41.05% val-macro:32.60% test-micro: 40.58% test-macro:28.84%
# channel: 6      END in 2097.71s--epoch:2492---val-micro: 41.92% val-macro:33.70% test-micro: 40.58% test-macro:29.69%
# channel: 8      END in 2150.17s--epoch:2185---val-micro: 42.50% val-macro:33.81% test-micro: 40.87% test-macro:29.84%
# channel: 10     END in 1585.46s--epoch:1306---val-micro: 42.06% val-macro:32.60% test-micro: 39.66% test-macro:27.63%
# channel: 12     END in 2091.64s--epoch:2010---val-micro: 41.56% val-macro:31.88% test-micro: 41.08% test-macro:30.20%
# ------the best is: 8, val-micro: 42.50%, val-macro: 33.81%tst-micro:40.87%, tst-macro:29.84%

    channel_list = [4,6,8,10,12]
    extra_args = ['--channel','-1','--kdim','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for channel in channel_list:
        extra_args[1] = channel
        extra_args[-1] = 128//channel
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('channel: %d'%channel,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = channel

    print('------the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%%, tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                                                                     ,best_test_f1[0]*100, best_test_f1[1]*100))

def test_ratio():
    #PPI
    # [Network]
# n_layer = 2
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45 
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 31

# [Optimizer]
# lr = 0.0057
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0  # 未使用

# [Run]
# epoch = 400
# early_stop = 1000
# ratio: 0.100000 END in 81.17s--epoch:149---val-micro: 18.86% val-macro:16.30% test-micro: 18.48% test-macro:15.34%
# ratio: 0.200000 END in 81.64s--epoch:179---val-micro: 20.44% val-macro:19.19% test-micro: 21.26% test-macro:17.37%
# ratio: 0.300000 END in 81.57s--epoch:240---val-micro: 21.73% val-macro:18.91% test-micro: 22.52% test-macro:19.25%
# ratio: 0.400000 END in 83.42s--epoch:200---val-micro: 23.36% val-macro:19.55% test-micro: 23.12% test-macro:20.59%
# ratio: 0.500000 END in 84.01s--epoch:397---val-micro: 24.41% val-macro:21.22% test-micro: 24.97% test-macro:21.25%
# ratio: 0.600000 END in 83.07s--epoch:248---val-micro: 24.88% val-macro:21.37% test-micro: 25.95% test-macro:21.31%
# ratio: 0.700000 END in 85.42s--epoch:358---val-micro: 27.68% val-macro:23.87% test-micro: 24.57% test-macro:20.62%
# ratio: 0.800000 END in 84.02s--epoch:304---val-micro: 26.51% val-macro:24.06% test-micro: 28.53% test-macro:24.00%
# ratio: 0.900000 END in 89.75s--epoch:250---val-micro: 27.11% val-macro:27.92% test-micro: 29.82% test-macro:26.49%

###POS

# [Network]
# n_layer = 6
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45 
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 31

# [Optimizer]
# lr = 0.051
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 1500
# early_stop = 1000
# ratio: 0.100000 END in 560.41s--epoch:1499---val-micro: 52.50% val-macro:22.50% test-micro: 52.56% test-macro:17.99%
# ratio: 0.200000 END in 572.72s--epoch:1039---val-micro: 53.75% val-macro:23.76% test-micro: 52.39% test-macro:23.88%
# ratio: 0.300000 END in 605.47s--epoch:1456---val-micro: 54.74% val-macro:28.52% test-micro: 52.19% test-macro:25.26%
# ratio: 0.400000 END in 642.89s--epoch:1493---val-micro: 54.38% val-macro:24.43% test-micro: 53.94% test-macro:26.97%
# ratio: 0.500000 END in 962.28s--epoch:1492---val-micro: 56.01% val-macro:30.15% test-micro: 54.86% test-macro:30.09%
# ratio: 0.600000 END in 968.08s--epoch:1462---val-micro: 55.56% val-macro:25.51% test-micro: 55.85% test-macro:35.79%
# ratio: 0.700000 END in 969.75s--epoch:1468---val-micro: 54.17% val-macro:33.51% test-micro: 56.85% test-macro:30.95%
# ratio: 0.800000 END in 970.88s--epoch:1356---val-micro: 57.57% val-macro:33.19% test-micro: 54.68% test-macro:30.60%
# ratio: 0.900000 END in 971.81s--epoch:1475---val-micro: 57.56% val-macro:38.02% test-micro: 57.10% test-macro:38.73%


# catablog
# ratio: 0.100000 END in 2293.89s--epoch:2640---val-micro: 34.31% val-macro:19.59% test-micro: 34.14% test-macro:19.20%
# ratio: 0.200000 END in 2585.88s--epoch:2877---val-micro: 36.84% val-macro:24.09% test-micro: 36.84% test-macro:23.81%

# 0、0078
# ratio: 0.100000 END in 2400.60s--epoch:1741---val-micro: 34.59% val-macro:20.83% test-micro: 33.86% test-macro:19.85%
# ratio: 0.200000 END in 2132.70s--epoch:1813---val-micro: 36.84% val-macro:25.25% test-micro: 36.23% test-macro:23.21%
# ratio: 0.300000 END in 1413.87s--epoch:1039---val-micro: 39.49% val-macro:26.77% test-micro: 37.76% test-macro:26.62%

# blogcatalog
# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.4
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0057
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 4000
# early_stop = 500
# ratio: 0.100000 END in 1698.15s--epoch:1371---val-micro: 34.05% val-macro:21.50% test-micro: 33.67% test-macro:19.38%
# ratio: 0.200000 END in 1709.57s--epoch:1350---val-micro: 37.17% val-macro:25.42% test-micro: 37.00% test-macro:22.85%
# ratio: 0.300000 END in 1323.88s--epoch:926---val-micro: 39.23% val-macro:27.25% test-micro: 37.90% test-macro:26.13%
# ratio: 0.400000 END in 1513.18s--epoch:1126---val-micro: 39.36% val-macro:28.40% test-micro: 37.19% test-macro:25.96%
# ratio: 0.500000 END in 1282.80s--epoch:870---val-micro: 38.49% val-macro:26.92% test-micro: 39.17% test-macro:28.02%
# ratio: 0.600000 END in 1222.00s--epoch:801---val-micro: 40.79% val-macro:29.58% test-micro: 38.12% test-macro:26.26%
# ratio: 0.700000 END in 1197.48s--epoch:774---val-micro: 40.30% val-macro:29.04% test-micro: 38.91% test-macro:28.24%
# ratio: 0.800000 END in 1272.07s--epoch:846---val-micro: 40.98% val-macro:33.09% test-micro: 37.87% test-macro:30.35%
# ratio: 0.900000 END in 1293.31s--epoch:857---val-micro: 40.32% val-macro:30.89% test-micro: 40.15% test-macro:30.81%
# ------the best is: 0.80, val-micro: 40.98%, val-macro: 33.09%tst-micro:37.87%, tst-macro:30.35%


# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.4
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.005
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 4000
# early_stop = 1500
# ratio: 0.100000 END in 3503.34s--epoch:2361---val-micro: 34.37% val-macro:21.84% test-micro: 33.75% test-macro:20.22%
# ratio: 0.200000 END in 3685.46s--epoch:2801---val-micro: 35.35% val-macro:26.78% test-micro: 35.37% test-macro:23.00%
# ratio: 0.300000 END in 2282.23s--epoch:982---val-micro: 39.37% val-macro:27.03% test-micro: 38.09% test-macro:26.57%
# ratio: 0.400000 END in 2309.56s--epoch:1006---val-micro: 39.69% val-macro:28.22% test-micro: 37.80% test-macro:26.02%
# 将regular改为0.00036
# ratio: 0.100000 END in 2690.74s--epoch:1518---val-micro: 34.39% val-macro:21.12% test-micro: 34.42% test-macro:19.33%
# ratio: 0.200000 END in 3672.07s--epoch:2677---val-micro: 37.14% val-macro:25.45% test-micro: 37.28% test-macro:23.72%
# ratio: 0.300000 END in 2798.73s--epoch:1539---val-micro: 39.79% val-macro:27.03% test-micro: 37.74% test-macro:27.18%
# ratio: 0.400000 END in 2182.67s--epoch:815---val-micro: 40.29% val-macro:28.60% test-micro: 38.53% test-macro:25.59%
# ratio: 0.500000 END in 2497.32s--epoch:1157---val-micro: 39.65% val-macro:27.55% test-micro: 40.01% test-macro:28.98%
# ratio: 0.600000 END in 2644.96s--epoch:1293---val-micro: 41.23% val-macro:31.60% test-micro: 39.38% test-macro:29.00%
# ratio: 0.700000 END in 2308.00s--epoch:961---val-micro: 41.21% val-macro:30.10% test-micro: 38.95% test-macro:28.19%
# ratio: 0.800000 END in 2274.31s--epoch:907---val-micro: 43.15% val-macro:33.37% test-micro: 41.01% test-macro:30.43%
# ratio: 0.900000 END in 2217.90s--epoch:851---val-micro: 41.19% val-macro:28.68% test-micro: 40.00% test-macro:29.73%
# ------the best is: 0.80, val-micro: 43.15%, val-macro: 33.37%tst-micro:41.01%, tst-macro:30.43%
# blogcatalog  0.5
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.4
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 37

# [Optimizer]
# lr = 0.0093
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00036
# clip = 2.0
# ratio: 0.100000 END in 2152.74s--epoch:1951---val-micro: 34.39% val-macro:20.62% test-micro: 32.72% test-macro:19.07%
# ratio: 0.200000 END in 1940.28s--epoch:1503---val-micro: 37.29% val-macro:24.90% test-micro: 37.48% test-macro:23.64%
# ratio: 0.300000 END in 1834.04s--epoch:1584---val-micro: 39.92% val-macro:26.73% test-micro: 38.03% test-macro:26.35%
# ratio: 0.400000 END in 1940.92s--epoch:1515---val-micro: 40.48% val-macro:28.56% test-micro: 38.64% test-macro:26.67%
# ratio: 0.500000 END in 1886.84s--epoch:1306---val-micro: 40.58% val-macro:30.10% test-micro: 41.40% test-macro:30.32%
# ratio: 0.600000 END in 1898.58s--epoch:1418---val-micro: 39.86% val-macro:28.94% test-micro: 40.42% test-macro:29.95%
# ratio: 0.700000 END in 1922.69s--epoch:937---val-micro: 42.14% val-macro:31.23% test-micro: 40.61% test-macro:30.67%
# ratio: 0.800000 END in 1928.22s--epoch:1229---val-micro: 40.37% val-macro:32.11% test-micro: 41.92% test-macro:31.42%
# ratio: 0.900000 END in 2007.70s--epoch:1452---val-micro: 43.82% val-macro:30.79% test-micro: 40.91% test-macro:30.37%

# train_ratio = 0.5

# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0093
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00036
# clip = 2.0

# [Run]
# epoch = 2000
# early_stop = 1000

# ratio: 0.100000 END in 1905.04s--epoch:1832---val-micro: 33.85% val-macro:19.85% test-micro: 33.72% test-macro:18.44%
# ratio: 0.200000 END in 1821.52s--epoch:1910---val-micro: 36.86% val-macro:24.50% test-micro: 36.58% test-macro:21.64%
# ratio: 0.300000 END in 1828.62s--epoch:1866---val-micro: 39.19% val-macro:26.44% test-micro: 38.51% test-macro:27.02%
# ratio: 0.400000 END in 1837.52s--epoch:1655---val-micro: 40.18% val-macro:29.13% test-micro: 38.72% test-macro:26.81%
# ratio: 0.500000 END in 1836.36s--epoch:1453---val-micro: 39.62% val-macro:27.38% test-micro: 41.26% test-macro:29.72%
# ratio: 0.600000 END in 1846.28s--epoch:1565---val-micro: 42.15% val-macro:32.37% test-micro: 39.73% test-macro:28.68%
# ratio: 0.700000 END in 1862.23s--epoch:1245---val-micro: 42.40% val-macro:31.42% test-micro: 39.67% test-macro:28.43%
# ratio: 0.800000 END in 1857.76s--epoch:1182---val-micro: 43.51% val-macro:33.70% test-micro: 40.37% test-macro:28.40%
# ratio: 0.900000 END in 1868.84s--epoch:1489---val-micro: 42.21% val-macro:28.21% test-micro: 42.26% test-macro:30.74%
# ------the best is: 0.80, val-micro: 43.51%, val-macro: 33.70%tst-micro:40.37%, tst-macro:28.40%

# 采用6层
# ratio: 0.100000 END in 2084.84s--epoch:1462---val-micro: 33.92% val-macro:18.65% test-micro: 33.73% test-macro:18.98%
# ratio: 0.200000 END in 2114.50s--epoch:1896---val-micro: 37.19% val-macro:24.14% test-micro: 36.79% test-macro:22.07%
# ratio: 0.300000 END in 2105.40s--epoch:1835---val-micro: 39.87% val-macro:26.00% test-micro: 38.43% test-macro:26.23%
# ratio: 0.400000 END in 2117.09s--epoch:1987---val-micro: 40.01% val-macro:29.44% test-micro: 39.09% test-macro:26.35%
# ratio: 0.500000 END in 2136.92s--epoch:1802---val-micro: 40.26% val-macro:27.58% test-micro: 40.93% test-macro:29.37%
# ratio: 0.600000 END in 2133.13s--epoch:1938---val-micro: 42.40% val-macro:31.61% test-micro: 40.61% test-macro:28.86%
# ratio: 0.700000 END in 2138.12s--epoch:1424---val-micro: 41.97% val-macro:31.35% test-micro: 39.86% test-macro:28.83%
# ratio: 0.800000 END in 2144.26s--epoch:1508---val-micro: 42.71% val-macro:33.64% test-micro: 39.37% test-macro:28.21%
# ratio: 0.900000 END in 2164.67s--epoch:1658---val-micro: 40.90% val-macro:30.27% test-micro: 42.26% test-macro:30.99%
# ------the best is: 0.80, val-micro: 42.71%, val-macro: 33.64%tst-micro:39.37%, tst-macro:28.21%

    ratio_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    extra_args = ['--train_ratio','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for ratio in ratio_list:
        extra_args[-1] = ratio
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('ratio: %f'%ratio,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = ratio

    print('------the best is: %.2f, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%%, tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                                                                     ,best_test_f1[0]*100, best_test_f1[1]*100))

def test_layer_dec():
    # 使用基层 POS
#     layer:2,deck:0  END in 148.99s--epoch:382---val-micro: 44.64% val-macro:8.58% test-micro: 42.01% test-macro:9.84%
# layer:2,deck:1  END in 149.49s--epoch:580---val-micro: 43.53% val-macro:7.02% test-micro: 42.31% test-macro:10.43%
# layer:2,deck:2  END in 148.14s--epoch:596---val-micro: 44.16% val-macro:7.64% test-micro: 42.77% test-macro:11.22%
# layer:2,deck:3  END in 160.90s--epoch:599---val-micro: 44.48% val-macro:8.29% test-micro: 43.07% test-macro:11.55%
# layer:2,deck:4  END in 152.02s--epoch:569---val-micro: 43.06% val-macro:6.00% test-micro: 41.70% test-macro:9.37%
# layer:2,deck:5  END in 148.76s--epoch:565---val-micro: 44.64% val-macro:8.02% test-micro: 43.38% test-macro:12.37%
# layer:2,deck:6  END in 161.14s--epoch:545---val-micro: 43.38% val-macro:6.38% test-micro: 41.89% test-macro:9.60%
# layer:2,deck:7  END in 185.61s--epoch:522---val-micro: 42.74% val-macro:5.53% test-micro: 41.70% test-macro:9.46%
# layer:2,deck:8  END in 155.21s--epoch:283---val-micro: 43.53% val-macro:6.90% test-micro: 41.70% test-macro:9.60%
# ---layer:2---the best is: 0, val-micro: 44.64%, val-macro: 8.58%tst-micro:0.00%, tst-macro:0.00%
# layer:3,deck:0  END in 176.77s--epoch:594---val-micro: 45.58% val-macro:10.18% test-micro: 43.68% test-macro:12.62%
# layer:3,deck:1  END in 179.30s--epoch:593---val-micro: 46.37% val-macro:11.00% test-micro: 44.75% test-macro:14.59%
# layer:3,deck:2  END in 178.23s--epoch:467---val-micro: 44.48% val-macro:7.88% test-micro: 42.47% test-macro:10.49%
# layer:3,deck:3  END in 186.53s--epoch:595---val-micro: 44.64% val-macro:8.09% test-micro: 43.07% test-macro:11.49%
# layer:3,deck:4  END in 227.97s--epoch:209---val-micro: 43.69% val-macro:7.05% test-micro: 42.47% test-macro:10.42%
# layer:3,deck:5  END in 223.93s--epoch:599---val-micro: 43.06% val-macro:5.95% test-micro: 42.16% test-macro:10.05%
# layer:3,deck:6  END in 226.60s--epoch:502---val-micro: 43.69% val-macro:6.87% test-micro: 42.31% test-macro:10.46%
# layer:3,deck:7  END in 224.79s--epoch:3---val-micro: 44.02% val-macro:5.81% test-micro: 42.73% test-macro:8.52%
# layer:3,deck:8  END in 223.55s--epoch:599---val-micro: 43.85% val-macro:7.01% test-micro: 43.26% test-macro:11.54%
# ---layer:3---the best is: 1, val-micro: 46.37%, val-macro: 11.00%tst-micro:0.00%, tst-macro:0.00%
# layer:4,deck:0  END in 269.49s--epoch:575---val-micro: 48.42% val-macro:19.06% test-micro: 44.60% test-macro:14.55%
# layer:4,deck:1  END in 269.23s--epoch:595---val-micro: 49.37% val-macro:20.22% test-micro: 46.12% test-macro:15.96%
# layer:4,deck:2  END in 266.49s--epoch:596---val-micro: 44.95% val-macro:9.35% test-micro: 43.38% test-macro:12.01%
# layer:4,deck:3  END in 261.38s--epoch:589---val-micro: 44.01% val-macro:7.70% test-micro: 43.07% test-macro:11.65%
# layer:4,deck:4  END in 258.18s--epoch:595---val-micro: 44.32% val-macro:7.79% test-micro: 42.92% test-macro:11.12%
# layer:4,deck:5  END in 193.10s--epoch:580---val-micro: 43.85% val-macro:7.32% test-micro: 42.01% test-macro:9.86%
# layer:4,deck:6  END in 177.91s--epoch:592---val-micro: 45.11% val-macro:9.04% test-micro: 43.56% test-macro:12.38%
# layer:4,deck:7  END in 176.40s--epoch:592---val-micro: 43.22% val-macro:6.19% test-micro: 42.01% test-macro:9.86%
# layer:4,deck:8  END in 176.55s--epoch:450---val-micro: 43.38% val-macro:6.61% test-micro: 41.86% test-macro:9.79%
# ---layer:4---the best is: 1, val-micro: 49.37%, val-macro: 20.22%tst-micro:0.00%, tst-macro:0.00%
# layer:5,deck:0  END in 208.63s--epoch:595---val-micro: 49.21% val-macro:19.42% test-micro: 46.58% test-macro:17.02%
# layer:5,deck:1  END in 205.01s--epoch:587---val-micro: 49.21% val-macro:17.04% test-micro: 46.42% test-macro:16.62%
# layer:5,deck:2  END in 202.69s--epoch:587---val-micro: 45.11% val-macro:10.77% test-micro: 43.53% test-macro:11.84%
# layer:5,deck:3  END in 197.28s--epoch:590---val-micro: 45.43% val-macro:9.65% test-micro: 42.92% test-macro:11.11%
# layer:5,deck:4  END in 196.02s--epoch:7---val-micro: 43.69% val-macro:5.88% test-micro: 41.74% test-macro:8.65%
# layer:5,deck:5  END in 194.12s--epoch:321---val-micro: 44.79% val-macro:8.45% test-micro: 42.92% test-macro:11.11%
# layer:5,deck:6  END in 193.48s--epoch:8---val-micro: 43.85% val-macro:6.77% test-micro: 41.89% test-macro:9.27%
# layer:5,deck:7  END in 193.39s--epoch:599---val-micro: 44.16% val-macro:7.79% test-micro: 43.38% test-macro:12.07%
# layer:5,deck:8  END in 193.34s--epoch:530---val-micro: 42.90% val-macro:5.73% test-micro: 42.31% test-macro:10.50%
# ---layer:5---the best is: 0, val-micro: 49.21%, val-macro: 19.42%tst-micro:0.00%, tst-macro:0.00%
# layer:6,deck:0  END in 229.08s--epoch:579---val-micro: 52.21% val-macro:22.03% test-micro: 48.59% test-macro:19.36%
# layer:6,deck:1  END in 225.49s--epoch:585---val-micro: 50.47% val-macro:21.74% test-micro: 47.07% test-macro:17.12%
# layer:6,deck:2  END in 222.15s--epoch:596---val-micro: 46.53% val-macro:11.15% test-micro: 44.14% test-macro:13.36%
# layer:6,deck:3  END in 229.27s--epoch:593---val-micro: 44.95% val-macro:9.11% test-micro: 43.68% test-macro:12.22%
# layer:6,deck:4  END in 258.85s--epoch:394---val-micro: 44.16% val-macro:7.92% test-micro: 42.62% test-macro:10.81%
# layer:6,deck:5  END in 318.06s--epoch:599---val-micro: 44.64% val-macro:11.21% test-micro: 42.50% test-macro:10.93%
#  layer:6,deck:6 END in 326.21s--epoch:592---val-micro: 45.11% val-macro:9.01% test-micro: 43.38% test-macro:12.14%
# layer:6,deck:7  END in 323.25s--epoch:7---val-micro: 43.38% val-macro:5.87% test-micro: 42.84% test-macro:9.12%
# layer:6,deck:8  END in 321.18s--epoch:502---val-micro: 43.85% val-macro:7.36% test-micro: 42.31% test-macro:10.28%
# ---layer:6---the best is: 0, val-micro: 52.21%, val-macro: 22.03%tst-micro:0.00%, tst-macro:0.00%
# layer:7,deck:0  END in 395.70s--epoch:557---val-micro: 52.05% val-macro:21.07% test-micro: 47.68% test-macro:18.41%
# layer:7,deck:1  END in 383.69s--epoch:592---val-micro: 49.05% val-macro:16.93% test-micro: 46.27% test-macro:16.27%
# layer:7,deck:2  END in 370.68s--epoch:596---val-micro: 48.26% val-macro:15.55% test-micro: 45.09% test-macro:14.40%
# layer:7,deck:3  END in 365.97s--epoch:588---val-micro: 45.74% val-macro:12.93% test-micro: 44.78% test-macro:14.00%
# layer:7,deck:4  END in 313.77s--epoch:565---val-micro: 44.32% val-macro:7.90% test-micro: 42.16% test-macro:10.62%
# layer:7,deck:5  END in 231.74s--epoch:578---val-micro: 44.64% val-macro:8.75% test-micro: 42.31% test-macro:10.53%
# layer:7,deck:6  END in 230.32s--epoch:433---val-micro: 43.69% val-macro:7.35% test-micro: 42.47% test-macro:10.72%
# layer:7,deck:7  END in 231.64s--epoch:427---val-micro: 44.48% val-macro:8.22% test-micro: 43.07% test-macro:11.51%
# layer:7,deck:8  END in 229.69s--epoch:599---val-micro: 43.22% val-macro:6.24% test-micro: 42.16% test-macro:10.33%
# ---layer:7---the best is: 0, val-micro: 52.05%, val-macro: 21.07%tst-micro:0.00%, tst-macro:0.00%
# layer:8,deck:0  END in 275.11s--epoch:593---val-micro: 52.84% val-macro:25.51% test-micro: 48.44% test-macro:19.54%
# layer:8,deck:1  END in 281.22s--epoch:598---val-micro: 48.90% val-macro:15.48% test-micro: 46.58% test-macro:17.15%
# layer:8,deck:2  END in 297.57s--epoch:595---val-micro: 47.16% val-macro:11.40% test-micro: 45.36% test-macro:15.22%
# layer:8,deck:3  END in 312.62s--epoch:590---val-micro: 44.48% val-macro:8.29% test-micro: 43.23% test-macro:11.60%
# layer:8,deck:4  END in 275.99s--epoch:224---val-micro: 45.27% val-macro:9.44% test-micro: 42.62% test-macro:10.65%
# layer:8,deck:5  END in 252.22s--epoch:597---val-micro: 43.85% val-macro:7.28% test-micro: 43.07% test-macro:11.88%
# blpgcatalog
# train_ratio = 0.8

# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0093
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00036
# clip = 2.0

# [Run]
# epoch = 3000
# early_stop = 1000
# layer:2,deck:0  END in 1327.37s--epoch:2393---val-micro: 42.50% val-macro:33.09% test-micro: 40.16% test-macro:28.27%
# layer:2,deck:1  END in 1242.25s--epoch:1855---val-micro: 42.06% val-macro:34.34% test-micro: 39.80% test-macro:29.87%
# layer:2,deck:2  END in 1292.75s--epoch:2330---val-micro: 42.57% val-macro:34.27% test-micro: 41.37% test-macro:30.75%
# layer:2,deck:3  END in 1269.12s--epoch:2270---val-micro: 42.71% val-macro:31.95% test-micro: 40.30% test-macro:28.66%
# layer:2,deck:4  END in 1253.68s--epoch:2703---val-micro: 41.13% val-macro:32.95% test-micro: 40.37% test-macro:29.10%
# layer:2,deck:5  END in 1234.03s--epoch:2116---val-micro: 43.00% val-macro:32.34% test-micro: 41.44% test-macro:29.24%
# ---layer:2---the best is: 2, val-micro: 42.57%, val-macro: 34.27%tst-micro:41.37%, tst-macro:30.75%
# layer:3,deck:0  END in 1674.55s--epoch:1963---val-micro: 42.14% val-macro:33.63% test-micro: 41.87% test-macro:30.52%
# layer:3,deck:1  END in 1638.13s--epoch:2055---val-micro: 42.71% val-macro:34.36% test-micro: 40.09% test-macro:29.45%
# layer:3,deck:2  END in 1588.25s--epoch:2117---val-micro: 42.14% val-macro:31.96% test-micro: 40.16% test-macro:30.50%
# layer:3,deck:3  END in 1535.46s--epoch:2972---val-micro: 42.14% val-macro:32.86% test-micro: 40.30% test-macro:28.36%
# layer:3,deck:4  END in 1480.80s--epoch:2707---val-micro: 42.42% val-macro:32.90% test-micro: 41.94% test-macro:30.85%
# layer:3,deck:5  END in 1436.03s--epoch:2988---val-micro: 41.56% val-macro:32.21% test-micro: 40.80% test-macro:29.06%
# ---layer:3---the best is: 1, val-micro: 42.71%, val-macro: 34.36%tst-micro:40.09%, tst-macro:29.45%
# layer:4,deck:0  END in 1581.03s--epoch:1306---val-micro: 42.06% val-macro:32.60% test-micro: 39.66% test-macro:27.63%
# layer:4,deck:1  END in 1954.34s--epoch:2252---val-micro: 41.77% val-macro:34.44% test-micro: 39.73% test-macro:29.44%
# layer:4,deck:2  END in 1843.27s--epoch:2959---val-micro: 42.78% val-macro:34.30% test-micro: 40.37% test-macro:30.25%
# layer:4,deck:3  END in 1184.01s--epoch:1059---val-micro: 41.27% val-macro:30.94% test-micro: 40.30% test-macro:28.24%
# layer:4,deck:4  END in 1676.48s--epoch:2735---val-micro: 40.91% val-macro:30.07% test-micro: 40.58% test-macro:29.15%
# layer:4,deck:5  END in 1580.43s--epoch:1905---val-micro: 41.92% val-macro:32.03% test-micro: 40.16% test-macro:29.41%
# ---layer:4---the best is: 2, val-micro: 42.78%, val-macro: 34.30%tst-micro:40.37%, tst-macro:30.25%
# ------the best is: (4,2), val-micro: 42.78%, val-macro: 34.30%tst-micro:40.37%, tst-macro:30.25%
    layer_list = [2,3,4]
    deck_list = [0,1,2,3,4,5]
    extra_args = ['--n_layer', '-1', '--deck', '-1']
    best_f1 = (0,0)  # 对于两个参数而言
    best_test_f1 = (0,0)
    best_hyper = (0,0)
    for layer in layer_list:
        best_layer_f1 = (0,0)
        best_layer_test_f1 = (0,0)
        best_layer_hyper = 0
        for deck in deck_list:
            extra_args[1] = layer
            extra_args[-1] = deck
            config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
            print('layer:%d,deck:%d'%(layer, deck),end='\t')
            val_f1, tst_f1= train.main(config)
            if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
                best_f1 = (val_f1['micro'], val_f1['macro'])
                best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
                best_hyper = (layer, deck)
            if sum((val_f1['micro'], val_f1['macro'])) > sum(best_layer_f1):
                best_layer_f1 = (val_f1['micro'], val_f1['macro'])
                best_layer_test_f1 = (tst_f1['micro'], tst_f1['macro'])
                best_layer_hyper = deck
        print('---layer:%d---the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                    'tst-micro:%.2f%%, tst-macro:%.2f%%'%(layer, best_layer_hyper, best_layer_f1[0]*100, best_layer_f1[1]*100\
                                                                                                        ,best_layer_test_f1[0]*100, best_layer_test_f1[1]*100))
            
    print('------the best is: (%d,%d), val-micro: %.2f%%, val-macro: %.2f%%'
                    'tst-micro:%.2f%%, tst-macro:%.2f%%'%(best_hyper[0], best_hyper[1], best_f1[0]*100, best_f1[1]*100\
                                                                                                        ,best_test_f1[0]*100, best_test_f1[1]*100))

def test_lr():
    lr_list = [0.011, 0.012,0.013,0.01,0.009,0.008,0.007,0.006]
    # POS
# n_layer = 6
# channel = 10
# kdim = 12
# deck = 0
# dropout = 0.45 
# routit = 6
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 31

# [Optimizer]
# lr = 0.051
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00059
# clip = 2.0

# [Run]
# epoch = 1500
# early_stop = 1000
# lr: 0.001130    END in 571.83s--epoch:1499---val-micro: 50.16% val-macro:15.06% test-micro: 46.88% test-macro:17.30%
# lr: 0.002384    END in 573.49s--epoch:1415---val-micro: 54.57% val-macro:27.74% test-micro: 50.88% test-macro:21.68%
# lr: 0.001570    END in 573.59s--epoch:1479---val-micro: 53.31% val-macro:21.47% test-micro: 47.95% test-macro:18.60%
# lr: 0.001582    END in 573.19s--epoch:1479---val-micro: 53.63% val-macro:21.75% test-micro: 48.40% test-macro:19.46%
# lr: 0.001587    END in 574.87s--epoch:1498---val-micro: 52.52% val-macro:20.86% test-micro: 47.64% test-macro:18.25%
# lr: 0.006116    END in 573.76s--epoch:1498---val-micro: 55.99% val-macro:28.92% test-micro: 51.49% test-macro:23.99%
# lr: 0.002901    END in 573.30s--epoch:1479---val-micro: 56.83% val-macro:32.25% test-micro: 51.79% test-macro:24.32%
# lr: 0.001050    END in 572.17s--epoch:1499---val-micro: 49.05% val-macro:13.83% test-micro: 47.18% test-macro:17.61%
# lr: 0.009091    END in 574.62s--epoch:1347---val-micro: 56.78% val-macro:29.78% test-micro: 51.49% test-macro:23.90%
# lr: 0.002177    END in 572.96s--epoch:1437---val-micro: 55.21% val-macro:27.82% test-micro: 50.88% test-macro:21.87%
# lr: 0.056718    END in 573.39s--epoch:1477---val-micro: 57.89% val-macro:32.56% test-micro: 54.53% test-macro:29.54%
# lr: 0.062074    END in 576.85s--epoch:1240---val-micro: 56.78% val-macro:32.05% test-micro: 51.52% test-macro:27.65%
# lr: 0.117364    END in 573.18s--epoch:1321---val-micro: 55.25% val-macro:34.48% test-micro: 53.01% test-macro:31.48%
# lr: 0.050810    END in 573.98s--epoch:1356---val-micro: 57.41% val-macro:33.39% test-micro: 52.86% test-macro:28.46%
# lr: 0.000929    END in 572.86s--epoch:1496---val-micro: 47.63% val-macro:12.12% test-micro: 46.42% test-macro:16.08%
# lr: 0.007460    END in 574.09s--epoch:1488---val-micro: 56.47% val-macro:29.22% test-micro: 51.64% test-macro:23.96%
# lr: 0.001940    END in 573.12s--epoch:1495---val-micro: 55.21% val-macro:25.51% test-micro: 50.88% test-macro:21.46%
# lr: 0.001762    END in 572.69s--epoch:1425---val-micro: 53.47% val-macro:21.66% test-micro: 50.11% test-macro:20.68%
# lr: 0.004302    END in 574.80s--epoch:1437---val-micro: 56.47% val-macro:28.31% test-micro: 52.40% test-macro:23.59%
# lr: 0.001153    END in 572.90s--epoch:1496---val-micro: 50.16% val-macro:14.93% test-micro: 47.64% test-macro:18.21%
# ------the best is: 0.050810, val-micro: 57.41%, val-macro: 33.39%tst-micro:52.86%, tst-macro:28.46%

# catblog 
# train_ratio = 0.8

# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 2
# dropout = 0.45
# routit = 5
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0093
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00036
# clip = 2.0

# [Run]
# epoch = 3000
# early_stop = 1000
# lr: 0.011000    END in 1690.41s--epoch:1836---val-micro: 42.06% val-macro:34.40% test-micro: 40.94% test-macro:28.93%
# lr: 0.012000    END in 2011.49s--epoch:1701---val-micro: 42.57% val-macro:34.13% test-micro: 40.58% test-macro:29.64%
# lr: 0.013000    END in 2806.79s--epoch:2721---val-micro: 42.50% val-macro:33.54% test-micro: 39.73% test-macro:28.71%
# lr: 0.010000    END in 2119.79s--epoch:2356---val-micro: 42.28% val-macro:34.63% test-micro: 41.58% test-macro:31.03%
# lr: 0.009000    END in 1765.96s--epoch:2172---val-micro: 42.71% val-macro:33.86% test-micro: 40.23% test-macro:29.82%
# lr: 0.008000    END in 2216.34s--epoch:1784---val-micro: 41.77% val-macro:32.21% test-micro: 39.87% test-macro:29.11%
# lr: 0.007000    END in 2800.77s--epoch:2704---val-micro: 41.99% val-macro:34.03% test-micro: 39.87% test-macro:30.11%
# ------the best is: 0.01, val-micro: 42.28%, val-macro: 34.63%tst-micro:41.58%, tst-macro:31.03%
    extra_args = ['--lr','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for lr in lr_list:
        extra_args[-1] = lr
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('lr: %f'%lr,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = lr

    print('------the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%%, tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                                                                     ,best_test_f1[0]*100, best_test_f1[1]*100))
def test_drop():
    # blogcatalog
# train_ratio = 0.8

# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 2
# dropout = 0.45
# routit = 5
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0093
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00036
# clip = 2.0

# [Run]
# epoch = 3000
# early_stop = 1000
# drop: 0.300000  END in 1317.64s--epoch:1231---val-micro: 41.85% val-macro:32.06% test-micro: 40.58% test-macro:29.58%
# drop: 0.350000  END in 1573.05s--epoch:1667---val-micro: 41.13% val-macro:32.46% test-micro: 39.51% test-macro:28.94%
# drop: 0.400000  END in 1698.31s--epoch:1883---val-micro: 42.28% val-macro:33.02% test-micro: 39.30% test-macro:28.61%
# drop: 0.450000  END in 1767.18s--epoch:2169---val-micro: 43.29% val-macro:33.60% test-micro: 40.94% test-macro:30.49%
# drop: 0.500000  END in 1764.01s--epoch:2646---val-micro: 41.56% val-macro:33.12% test-micro: 40.87% test-macro:30.56%
# ------the best is: 0.45, val-micro: 43.29%, val-macro: 33.60%tst-micro:40.94%, tst-macro:30.49%
    drop_list = [0.30, 0.35, 0.40, 0.45, 0.50]
    extra_args = ['--dropout','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for drop in drop_list:
        extra_args[-1] = drop
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('drop: %f'%drop,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = drop

    print('------the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%%, tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                                                                     ,best_test_f1[0]*100, best_test_f1[1]*100))


def test_reg():
        # blogcatalog
# train_ratio = 0.8

# [Network]
# n_layer = 4
# channel = 10
# kdim = 12
# deck = 2
# dropout = 0.45
# routit = 5
# tau =1.0
# nbsz = 32
# include_self = 0
# threshold = 0.6
# resample = 1
# jump= 1
# seed = 666

# [Optimizer]
# lr = 0.0093
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8
# reg = 0.00036
# clip = 2.0

# [Run]
# epoch = 3000
# early_stop = 1000

# reg: 0.001000   END in 1989.64s--epoch:1706---val-micro: 39.03% val-macro:31.30% test-micro: 38.23% test-macro:29.33%
# reg: 0.005000   END in 1262.09s--epoch:1138---val-micro: 32.76% val-macro:22.07% test-micro: 29.96% test-macro:16.32%
# reg: 0.000360   END in 1768.56s--epoch:2169---val-micro: 43.29% val-macro:33.60% test-micro: 40.94% test-macro:30.49%
# reg: 0.000450   END in 1640.08s--epoch:1781---val-micro: 41.56% val-macro:33.12% test-micro: 39.44% test-macro:28.92%
# reg: 0.000260   END in 1768.97s--epoch:2351---val-micro: 42.21% val-macro:32.94% test-micro: 40.66% test-macro:30.10%
# reg: 0.000550   END in 1444.56s--epoch:1454---val-micro: 41.27% val-macro:32.98% test-micro: 39.30% test-macro:29.17%
# reg: 0.000150   END in 1766.35s--epoch:2966---val-micro: 42.14% val-macro:33.86% test-micro: 40.09% test-macro:28.13%
# ------the best is: 0, val-micro: 43.29%, val-macro: 33.60%tst-micro:40.94%, tst-macro:30.49%
    reg_list = [0.001,0.005,0.00036,0.00045,0.00026,0.00055,0.00015,]
    extra_args = ['--reg','-1']
    best_f1 = (0,0)
    best_test_f1 = (0,0)
    best_hyper = -1
    for reg in reg_list:
        extra_args[-1] = reg
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        print('reg: %f'%reg,end='\t')
        val_f1, tst_f1= train.main(config)
        if sum((val_f1['micro'], val_f1['macro'])) > sum(best_f1):
            best_f1 = (val_f1['micro'], val_f1['macro'])
            best_test_f1 = (tst_f1['micro'], tst_f1['macro'])
            best_hyper = reg

    print('------the best is: %d, val-micro: %.2f%%, val-macro: %.2f%%'
                'tst-micro:%.2f%%, tst-macro:%.2f%%'%(best_hyper, best_f1[0]*100, best_f1[1]*100\
                                                                                                     ,best_test_f1[0]*100, best_test_f1[1]*100))
def test_four():
    lr_list =  np.exp(np.random.uniform(-7,-3,size=20)).tolist()
    drop_list = [0.40,0.45,0.35,0.50]
    reg_list = np.exp(np.random.uniform(-9,-4,size=10)).tolist()
    rout_list = [5,6,7,8]
    channel_list = [7,8,9,10,11,12]
    extra_args = ['--lr','-1','--dropout','-1','--reg','-1','--routit','-1','channel', '-1','--kdim','-1']
    result_list = []
    trail_max = 100
    for trail in range(trail_max):
        lr = np.random.choice(lr_list, 1).tolist()[0]
        drop = np.random.choice(drop_list, 1).tolist()[0]
        reg = np.random.choice(reg_list,1).tolist()[0]
        rout = np.random.choice(rout_list,1,p=[0.3,0.3,0.2,0.2]).tolist()[0]
        channel = np.random.choice(channel_list,1,p=[0.1,0.2,0.1,0.3,0.1,0.2]).tolist()[0]
        k_dim = 128 // channel
        extra_args[1] =lr
        extra_args[3] =drop
        extra_args[5] = reg
        extra_args[7] =rout
        extra_args[9] = channel
        extra_args[11] =k_dim
        config = Configurable('/home/wqt/code/DisenGCN/DisenGCN-multi/dataset/default.cfg', extra_args)
        val_f1, tst_f1= train.main(config)
        sample_dict = {}
        sample_dict['lr'] = lr
        sample_dict['drop'] = drop
        sample_dict['reg'] = reg
        sample_dict['rout'] = rout
        sample_dict['channel'] = channel
        sample_dict['k_dim'] =k_dim
        sample_dict['val'] = val_f1
        sample_dict['tst'] = tst_f1
        result_list.append(sample_dict)
    with open('blog_result.json', 'w') as fout:
        json.dump(result_list, fout)

# test_rout_it()
# test_c_h()
# test_layer_dec()
# test_lr()
# test_reg()
# test_drop()
# test_ratio()
test_four()
# 测试一下，找到更多的可能