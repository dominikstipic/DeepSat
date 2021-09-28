from pprint import pprint
import functools
from pathlib import Path 

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

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
        self.name = "metrics.csv" if not when else f"metrics-{when}.csv"
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

class ModelSaver(Subscriber):
    def __init__(self, path: str, buffer_size: int, when=None):
        Subscriber.__init__(self, when)
        self.dir_path = path
        pipeline_repository.create_dir_if_not_exist(self.dir_path)
        self.buffer_size = buffer_size

    def update(self, model_state_dict, epoch, **kwargs):
        epoch = epoch % self.buffer_size
        path = self.dir_path + f"/{epoch}.pt"
        path = pipeline_repository.get_path(path)
        torch.save(model_state_dict, str(path))

##################################################
    
class _EarlyStopperError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)

class EarlyStoper(Subscriber):
    def __init__(self, lookback, when=None):
        Subscriber.__init__(self, when)
        assert lookback >= 2, "lookback parameter must be greater than 2"
        self.lookback = lookback
        self.losses = []
    
    def is_monothonicaly_increasing(self, xs):
        values = []
        for i in range(len(xs)-1):
            increasing = xs[i+1] > xs[i]
            values.append(increasing)
        value = functools.reduce(lambda a, b: a and b, values)
        return value

    def last_k(self, xs, k):
        return xs[-k:]

    def update(self, loss, **kwargs):
        self.losses.append(loss)
        if len(self.losses) < self.lookback:
            return
        rev_losses = self.last_k(self.losses, self.lookback)
        value = self.is_monothonicaly_increasing(rev_losses)
        if value:
            raise _EarlyStopperError(f"Early stoping at epoch {len(self.losses)}")

##################################################

class StepPredictionSaver(Subscriber):
    def __init__(self, path: str, period: int, when=None):
        Subscriber.__init__(self, when)
        self.path = Path(path)
        self.period = period
    
    def update(self, prediction, iteration, **kwargs):
        if iteration % self.period != 0:
            return
        dir_name = Path("evaluation/artifacts/predictions")
        dir_name = pipeline_repository.create_dir_if_not_exist(dir_name)    
        for one_image in prediction:
            pil = to_pil_image(one_image.float())
            img_path = dir_name / f"{iteration}.png"
            pil.save(img_path)

