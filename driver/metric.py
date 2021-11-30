from sklearn.metrics import *
def eval_pr(pred_score, pred_label, true_label):
    pred_score = pred_score.reshape(-1)
    pred_label = pred_label.reshape(-1)
    true_label = true_label.reshape(-1)
    acc = accuracy_score(true_label, pred_label)
    p = precision_score(true_label,pred_label)
    r = recall_score(true_label, pred_label)
    if (p+r) != 0:
        f1 = 2*p*r/(p+r)
    else:
        f1 = 0
    auc = roc_auc_score(true_label, pred_score)
    return {'acc': acc,
             'p': p,
             'r': r,
             'f1':f1,
             'auc': auc}








    

