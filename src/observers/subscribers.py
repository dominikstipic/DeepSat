from pathlib import Path
import shutil
from pprint import pprint

import numpy as np
import torch
import matplotlib.pyplot as plt

import src.utils.pipeline_repository as pipeline_repository


class Subscriber():
    def __init__(self, when):
        self.when = when

    def update(self, **kwargs):
        pass
    
    def get(self):
        pass

    def reset_state(self):
        pass

##################################################   

class Confusion_Matrix(Subscriber):
    def __init__(self, class_num: int, metrics: list, when=None, ignore_idx=[]):
        if type(ignore_idx) == int: ignore_idx = [ignore_idx]
        Subscriber.__init__(self, when)
        self.CF = np.zeros([class_num, class_num])
        self.class_num = class_num
        self.ignore_idx = ignore_idx
        self._observers = {}
        for metric in metrics:
            self._observers[metric.__name__] = metric

    @property
    def observers(self):    
        self._observers

    def add_observer(self, name, func):    
        print(f"metric {name} subscribed to the confusion matrix")
        self._observers[name] = func

    def calc_conf_matrix(self, y_pred, labels, class_num, ignore_idx = []):
        cf = np.zeros((class_num, class_num), dtype=np.uint64)
        for i in range(class_num):
            for j in range(class_num):
                s = torch.logical_and(y_pred == i, labels == j).sum().item()
                if j in ignore_idx: continue
                cf[i,j] = s
        return cf

    def update(self, prediction, target, **kwargs):
        cf = self.calc_conf_matrix(prediction, target, self.class_num, self.ignore_idx)
        self.CF += cf
    
    def reset_state(self):
        self.CF = np.zeros([self.class_num, self.class_num])

    def __str__(self):
        return str(self.CF)

    def get(self):
        to_return = {}
        for name,func in self._observers.items():
            v = func(self.CF)
            to_return[name] = v
        return to_return

##################################################

class Running_Loss(Subscriber):
    def __init__(self, name, when=None):
        Subscriber.__init__(self, when)
        self.name = name
        self.loss = 0
        self.n = 0

    def update(self, input, loss, **kwargs):
        batch_num = len(input)
        batch_loss_sum = loss * batch_num
        self.n += batch_num
        self.loss += batch_loss_sum  

    def __str__(self):
        d = self.get()
        return str(d)

    def reset_state(self):
        self.loss = 0
        self.n = 0

    def get(self):
        avg_loss = None if self.n == 0 else (self.loss / self.n).item()
        d = {self.name : avg_loss}
        return d

##################################################


class StdPrinter(Subscriber):
    def __init__(self, when=None):
        Subscriber.__init__(self, when)

    def update(self, epoch, metrics, **kwargs):
        print(f"epoch : {epoch}")
        pprint(metrics)
        print("*******")


##################################################


class MetricSaver(Subscriber):
    def __init__(self, path: str, when=None):
        Subscriber.__init__(self, when)
        self.path = pipeline_repository.get_path(path)
        self.name = "metrics.csv"
        self.first_write = True


    def update(self, metrics, epoch, **kwargs):
        metrics["epoch"] = epoch
        xs = list(metrics.items())
        csv_header = list(map(lambda x: x[0], xs))
        data = list(map(lambda x: x[1], xs)) 
        append = True
        if self.first_write:
            append = False
            self.first_write = False
        pipeline_repository.push_csv(self.path, self.name, csv_header, [data], append=append)
            


##################################################

# class PredictionSaver(Subscriber):
#     n = 1
#     figsize = (10,10)
#     def __init__(self, root, period):
#         Subscriber.__init__(self)
#         self.period = period
#         self.root = Path(root) / "predictions"
#         if self.root.exists():
#             shutil.rmtree(self.root)
#         self.root.mkdir()

#     def update(self, kwargs):
#         if "prediction" not in kwargs.keys():
#             raise "Dictionary doesn't contain needed keys"
#         epoch, iteration = kwargs["epoch"], kwargs["iter"]
#         if iteration % self.period == 0:
#             y_pred = kwargs["prediction"]
#             name = f"epoch {epoch}, iter {iteration}.png"
#             y_pred = y_pred.cpu()
#             self.save(name, y_pred)
#         self.n += 1
    
#     def save(self, name, y_pred):
#         batch_size = len(y_pred)
#         nrow = int(np.sqrt(batch_size))
#         ncol = batch_size - nrow
#         plt.figure(figsize=self.figsize)
#         if nrow > 0 and ncol > 0:
#             for k in range(nrow+ncol):
#                 plt.subplot(nrow, ncol, k+1)
#                 plt.imshow(y_pred[k])
#                 plt.axis("off")
#             plt.tight_layout()
#         else:
#             plt.imshow(y_pred.squeeze())
#             plt.axis("off")
#         name = self.root / name
#         plt.savefig(name)
#         plt.close()

#     def reset_state(self):
#         self.n = 1

            

