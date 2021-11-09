from pprint import pprint
import functools
from pathlib import Path 
import operator

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
from PIL import ImageFont

import src.utils.pipeline_repository as pipeline_repository
from src.utils.common import h_concatenate_images, renorm_tensor

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
        self.CF = np.zeros([class_num, class_num], dtype=np.ulonglong)
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

    def update(self, epoch, metrics, state, **kwargs):
        print(f"state: {state}, epoch: {epoch}")
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
    
    def update(self, prediction, iteration, target, **kwargs):
        if iteration % self.period != 0:
            return
        dir_name = Path("evaluation/artifacts/predictions")
        dir_name = pipeline_repository.create_dir_if_not_exist(dir_name)    
        for pred, label in zip(prediction, target):
            pred, label = to_pil_image(pred.float()), to_pil_image(label.float())
            image = h_concatenate_images(pred, label)
            img_path = dir_name / f"{iteration}.png"
            image.save(img_path)

##################################################

class ChosenK(Subscriber):
    def __init__(self, path: str, k: int, is_worst_k: bool, when=None):
        Subscriber.__init__(self, when)
        self.path = pipeline_repository.create_dir_if_not_exist(Path(path))
        self.k = k
        self.list = []
        self.compar = operator.lt if is_worst_k else operator.gt
        self.is_worst_k = is_worst_k
        
    def create_image(self, input, prediction, target, loss, norm_mean, norm_std):
        input, prediction, target = input.squeeze(), prediction.squeeze(), target.squeeze() 
        input = renorm_tensor(input, norm_mean, norm_std)
        input, prediction, target = to_pil_image(input), to_pil_image(prediction.float()), to_pil_image(target.float()) 
        fig = h_concatenate_images(input, target)
        fig = h_concatenate_images(fig, prediction)

        font = ImageFont.load_default()
        draw = ImageDraw.Draw(fig)
        draw.text((25, 25), f"loss={round(loss.item(),3)}", font=font, fill=(255, 0, 0))
        return fig
    
    def get_ref(self):
        best_idx, ref_loss = 0, np.inf
        for i, (_, loss) in enumerate(self.list):
            if self.compar(loss, ref_loss):
                ref_loss = loss
                best_idx  = i
        return best_idx, ref_loss
    
    def get_path(self, iteration):
        return self.path / f"iteration-{iteration}.png"

    def insert(self, fig, loss, iteration):
        self.list.append([fig, loss, iteration])
        self.list = sorted(self.list, key=lambda x: x[1])
        if not self.is_worst_k:
            self.list = list(reversed(self.list))
    
    def delete_best(self, path):
        _,_,iteration = self.list[0]
        del self.list[0]
        path = self.get_path(iteration)
        path.unlink()

    def update(self, input, prediction, target, iteration, loss, norm_mean, norm_std, **kwargs):
        path = self.get_path(iteration)
        if len(self.list) < self.k:
            fig = self.create_image(input, prediction, target, loss, norm_mean, norm_std)
            self.insert(fig, loss, iteration)
            fig.save(str(path))
            return
        fig = self.create_image(input, prediction, target, loss, norm_mean, norm_std)
        _, ref_loss,_ = self.list[0]
        if loss < ref_loss:
            self.delete_best(path)
            self.insert(fig, loss, iteration)
            fig.save(str(path))

##################################################

class WorstK(ChosenK):
    def __init__(self, path: str, k: int, when=None):
        ChosenK.__init__(self, path=path, k=k, is_worst_k=True, when=when)
        
class BestK(ChosenK):
    def __init__(self, path: str, k: int, when=None):
        ChosenK.__init__(self, path=path, k=k, is_worst_k=False, when=when)

