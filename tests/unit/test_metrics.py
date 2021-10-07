import torch
import numpy as np

import src.observers.subscribers as subscribers
import src.observers.metrics as metrics

def get_prediction_and_mask():
    X = [[0,0,0,0], 
         [0,0,0,0], 
         [0,1,1,1], 
         [0,1,1,1]]
    Y = [[0,1,1,1],
         [0,0,1,1],
         [0,0,1,1],
         [0,0,1,1]]
    X, Y = torch.tensor(X), torch.tensor(Y) 
    return X, Y

TP = 4
FP = 2
TN = 5
FN = 5

def get_confusion_matrix():
    metrics_list = [metrics.accuracy, 
                    metrics.recall, 
                    metrics.precission, 
                    metrics.mIoU]
    cf = subscribers.Confusion_Matrix(class_num=2, metrics=metrics_list)
    return cf

def get_cf():
    cf = get_confusion_matrix()
    X,Y = get_prediction_and_mask()
    cf.update(prediction=X, target=Y)
    result = cf.CF
    return result

def test_confussion_matrix():
    result = get_cf()
    target = [[TN, FN],
              [FP, TP]]
    target = np.array(target)
    assert (result == target).all()

def test_accuracy():
    target = TP/(TP+FP+TN+FN)
    cf = get_cf()
    res = cf[1,1]/cf.sum()
    assert (res == target).all()

def test_precission():
    target = TP/(TP+FP)
    cf = get_cf()
    res = cf[1,1]/(cf[1,1] + cf[1,0])
    assert (res == target).all()

def test_recall():
    target = TP/(TP+FN)
    cf = get_cf()
    res = cf[1,1]/(cf[1,1] + cf[0,1])
    assert (res == target).all()

def test_miou():
    target = TP/(TP+FP+FN)
    cf = get_cf()
    res = cf[1,1]/(cf[1,1] + cf[0,1] + cf[1,0])
    assert (res == target).all()