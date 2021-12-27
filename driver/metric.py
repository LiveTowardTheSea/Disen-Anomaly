from sklearn.metrics import *
def eval_pr(pred_score, pred_label, true_label, anomaly_idx):
    pred_score = pred_score.reshape(-1)
    pred_label = pred_label.reshape(-1)
    true_label = true_label.reshape(-1)
    anomaly_idx = anomaly_idx.reshape(-1)
    # 总共rank 是多少
    max_anom_num = anomaly_idx.shape[0]
    rank_list = [300, 200, 100, 50]
    assert rank_list[0] == max_anom_num, 'rank_list inconsistency happen '
    p_list = []
    change_num = 0
    for i, rank in enumerate(rank_list):
        if i != 0:
            change_num += rank_list[i-1] - rank_list[i]
            change_anom_idx = anomaly_idx[-change_num:]   # 从这个往后
            pred_label[change_anom_idx]  = 0
    # acc = accuracy_score(true_label, pred_label)
        p = precision_score(true_label, pred_label)
        p_list.append(p)
        # r = recall_score(true_label, pred_label)
        # if (p+r) != 0:
        #     f1 = 2*p*r/(p+r)
        # else:
        #     f1 = 0
    auc = roc_auc_score(true_label, pred_score)
    return {
             'p': p_list,
            #  'r': r,
            #  'f1':f1,
            # 'acc': acc,
             'auc': auc
            }








    

