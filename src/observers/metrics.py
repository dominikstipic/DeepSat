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

def binary_cf(cf, type="macro"):
    if cf.shape == (2,2): 
        return cf
    else:
        raise RuntimeError("Not implemented! Implement it")

def accuracy(cf):
    cf = binary_cf(cf)
    correct = cf.trace()
    total_size = int(cf.sum())
    avg_pixel_acc = _safe_div(correct, total_size)
    return avg_pixel_acc

def recall(cf):
    cf = binary_cf(cf)
    TP = cf[1, 1]
    FN = cf[0, 1]
    recall = _safe_div(TP, TP+FN) 
    return recall

def precission(cf):
    cf = binary_cf(cf)
    TP = cf[1, 1]
    FP = cf[1, 0]
    precission = _safe_div(TP, TP+FP)
    return precission

def mIoU(cf):
    cf = binary_cf(cf)
    TP = cf[1, 1]
    FN = cf[0, 1]
    FP = cf[1, 0]
    iou =_safe_div(TP, TP + FN + FP)
    return iou

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


