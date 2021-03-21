import pickle
import gzip
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
import torch
import numpy as np
import pandas as pd


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])[0]


def return_metrics(true, preds):
    preds = torch.cat(preds, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()
    thres = find_optimal_cutoff(true, preds)
    preds_label = np.where(preds > thres, 1, 0)
    bas = balanced_accuracy_score(true, preds_label)
    auc = roc_auc_score(true, preds)
    return bas, auc, thres


def load_checkpoint(cfg, name):
    model_path = "{}/{}.ckpt".format(cfg.model_dir, name)
    checkpoint = torch.load(model_path, map_location="cuda" if cfg.use_cuda else "cpu")
    return checkpoint
