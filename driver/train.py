import numpy as np
import torch
import random
import sys
sys.path.append('/home/wqt/code/DisenGCN/Disen-Anomaly')
from config import Configurable
import argparse
import numpy as np
from DisenHelper import DisenHelper
from data.data_reader import DataLoader
from torch.optim.lr_scheduler import *
import os
import time
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter,lr=config.lr, betas=(config.beta1,config.beta2),eps=config.epsilon,weight_decay=config.reg)
        self.schedular = StepLR(self.optim, step_size=10, gamma=0.7)

    def step(self):
        self.optim.step()
        self.schedular.step()
        self.optim.zero_grad()

# 在macro偶然出现的很高的情况，实则并不具备泛化能力
# 在偶然运行的两次发现的

def train(classifier, config):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, classifier.model.parameters()), config)
    classifier.initialize_model()
    # 用于绘图
    loss_list = []
    metric_list = []
    best_f1 = 0
    best_epoch = 0
    wait_cnt = 0
    sts_time = time.time()
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    init_metric = classifier.get_metric()
    print('Epoch: 0/ ------ p-50:{:.2f}%, p-100:{:.2f}%, p-200:{:.2f}%, p-300:{:.2f}%,' 
          'structure_num:{}, attribute_num:{}, roc-auc:{:.2f}% '.format(init_metric['p'][-1]*100, init_metric['p'][-2]*100, init_metric['p'][-3]*100, init_metric['p'][-4]*100,
                                                init_metric['structure_num'], init_metric['attribute_num'], init_metric['auc']*100))
    for i in range(config.epoch):
        loss_dict = classifier.compute_loss()
        loss_value = loss_dict['loss']
        loss_value.backward()
        loss_item = loss_value.detach().cpu().item()
        adj_loss_item = loss_dict['adj_loss'].detach().cpu().numpy()
        att_loss_item = loss_dict['att_loss'].detach().cpu().numpy()
        loss_list.append(loss_item)
        # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, classifier.model.parameters()), \
        #                 max_norm=config.clip)
        optimizer.step()
        metrics = classifier.get_metric()
        metric_list.append(metrics)
        print('Epoch: {}/{} ---- train-loss:{:.2f}, adj_loss:{:.2f}, att_loss:{:.2f}, p-50:{:.2f}%, '
               'p-100:{:.2f}%, p-200:{:.2f}%, p-300:{:.2f}%, structure_num:{}, attribute_num:{}, roc-auc:{:.2f}% '.format(i+1,
                                                        config.epoch, loss_item, adj_loss_item, att_loss_item,
                                                        metrics['p'][-1]*100, metrics['p'][-2]*100, metrics['p'][-3]*100, metrics['p'][-4]*100,
                                                        metrics['structure_num'], metrics['attribute_num'], metrics['auc']*100))
        
        if metrics['auc'] >= best_f1:
            wait_cnt = 0
            best_f1 = metrics['auc']
            best_epoch = i+1
            torch.save(classifier.model.state_dict(), config.save_model_path)
        else:
            wait_cnt += 1
            if wait_cnt == config.early_stop:
                break
    #set_random_seed(79)   
    classifier.initialize_model(load_model_path=config.load_model_path)
    
    test_m = classifier.get_metric(print_detail=True)
    end_time = time.time()
    print('Traning finished, epoch:{}, best-result: p-50:{:.2f}%, p-100:{:.2f}%, p-200:{:.2f}%, p-300:{:.2f}%, '
          'structure_num:{}, attribute_num:{}, roc-auc:{:.2f}% '.format(best_epoch,
                                                        test_m['p'][-1]*100, test_m['p'][-2]*100, test_m['p'][-3]*100, test_m['p'][-4]*100,
                                                        test_m['structure_num'], test_m['attribute_num'], test_m['auc']*100))
    #print('%.4f %.4f'%(best_acc, tst_acc))
    # np.save('driver/plot/loss', np.array(loss_list))
    # np.save('driver/plot/micro', np.array(micro_list))
    # np.save('driver/plot/macro', np.array(macro_list))
    return test_m

def main(config=None):
    # 关于模型参数的配置
    gpu = torch.cuda.is_available()
    #print('gpu available: ', gpu)

    opt = argparse.ArgumentParser()
    opt.add_argument('--config_file',default='dataset/default.cfg')
    opt.add_argument('--cpu', action='store_true')
    args, extra_args = opt.parse_known_args()
    if config is None:
        config = Configurable(args.config_file, extra_args)
    set_random_seed(config.seed)
    config.use_cuda = False
    if gpu and not args.cpu:
        config.use_cuda = True

    Data_reader = DataLoader(config)  # 数据读入
    Disen_Classifier = DisenHelper(Data_reader, config)
    metric = train(Disen_Classifier, config)
    return metric
   
if __name__=='__main__':
    main()