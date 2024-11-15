import numpy as np
import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def com_sim(y_true, y_pred, k=10):
    all_rel = []
    for i in range(len(y_true)):
        gt_embd = y_true[i] # N_gt, D
        rec_embd = y_pred[i] # K, D
        
        if k == -1:
            rec_embd = rec_embd[:len(gt_embd)]
        else:
            rec_embd = rec_embd[:k]
        
        gt_embd = gt_embd / np.linalg.norm(gt_embd, axis=1, keepdims=True)
        rec_embd = rec_embd / np.linalg.norm(rec_embd, axis=1, keepdims=True)
        
        qd = np.dot(rec_embd, gt_embd.T)
        
        maxs = np.max(qd, axis=1)
        rel = np.mean(maxs)
        all_rel.append(rel)
    return np.mean(all_rel)

def max_sim(y_true, y_pred):
    # y_true: N_gt, D
    # y_pred: 1, D
    all_maxs = []
    for i in range(len(y_true)):
        gt_embd = y_true[i] # N_gt, D
        rec_embd = y_pred[i] # K, D
        
        gt_embd = gt_embd / np.linalg.norm(gt_embd, axis=1, keepdims=True)
        rec_embd = rec_embd / np.linalg.norm(rec_embd, axis=1, keepdims=True)
        
        qd = np.dot(rec_embd, gt_embd.T)
        
        maxs = np.max(qd, axis=1)
       
        all_maxs.append(maxs)
    return np.array(all_maxs).squeeze()