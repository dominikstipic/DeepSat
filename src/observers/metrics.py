import sys
import functools
import inspect

import  numpy as np

def _safe_div(x,y):
    return x/y if y != 0 else np.nan 

def _all_metrics_as_dict():
    results = {}
    module_dict = dict(inspect.getmembers(sys.modules[__name__]))
    for key, value in module_dict.items():
        if not key.startswith("_"):
            results[key] = value
    return results

def accuracy(cf):
    correct = cf.trace()
    total_size = int(cf.sum())
    avg_pixel_acc = _safe_div(correct, total_size)
    return avg_pixel_acc

def recall(cf):
    recalls = np.zeros(len(cf))
    for i in range(len(cf)):
        TP = cf[i,i]
        FN = cf[:,i].sum() - TP
        recall = _safe_div(TP, TP+FN) 
        recalls[i] = recall
    recall = recalls.mean()
    return recall.item()

def precission(cf):
    precissions = np.zeros(len(cf))
    for i in range(len(cf)):
        TP = cf[i,i]
        FP = cf[i].sum() - TP
        precission = _safe_div(TP, TP+FP)
        precissions[i] = precission
    precission = precissions.mean()
    return precission.item()

def mIoU(cf):
    ious = np.zeros(len(cf))
    for i in range(len(cf)):
        TP = cf[i,i]
        FP = cf[i].sum() - TP
        FN = cf[:,i].sum() - TP
        ious[i] = _safe_div(TP, TP+FP+FN)
    mean_iou = ious.mean().item()
    return mean_iou

def mIoU_per_class(class_info):
    @functools.wraps(class_info)
    def wrapper(cf):
        assert len(cf) == len(class_info)
        ious = {}
        for i in range(len(cf)):
            name = class_info[i]
            TP = cf[i,i]
            FP = cf[i].sum() - TP
            FN = cf[:,i].sum() - TP
            ious[name] = _safe_div(TP, TP+FP+FN)
        return ious
    return wrapper

METRIC_FUNS = _all_metrics_as_dict()


