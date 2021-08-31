import os
from pathlib import Path
import shutil

import torch
from tqdm import tqdm
import numpy as np

from src.utils.utils import write_json, read_json

class Storage:
    _instance    = None
    _storage_dir = None
    
    @classmethod
    def __init__(cls, storage_dir):
        cls._storage_dir = storage_dir

    @classmethod
    def get(cls):
        if not cls._instance:
            cls._instance = StorageBase(cls._storage_dir)
            #cls._instance.clear()
        return cls._instance

class StorageBase:
    latest_epoch = -1
    def __init__(self, storage_dir):
        self.storage_dir = Path(storage_dir)
        self.weights_path = self.storage_dir / "weights"
        self.metric_path  = self.storage_dir / "metrics"
        if not self.weights_path.exists():
            self.weights_path.mkdir()
        if not self.metric_path.exists():
            self.metric_path.mkdir()

    def size(self):
        epochs = self.get_epochs()
        return len(epochs)
    
    def paths_from_epoch(self, epoch):
        model_name  = self.weights_path  / f"{epoch}"
        metric_name = self.metric_path / f"{epoch}.json"
        return model_name, metric_name 
    
    def get_metrics(self):
        metrics = []
        epochs = self.get_epochs()
        for epoch in epochs:
            _, metric_path = self.paths_from_epoch(epoch)
            metric = read_json(metric_path)
            metrics.append(metric)
        return metrics
        
    def get_epochs(self):
        epochs = list(self.weights_path.iterdir())
        epochs = [e.parts[-1] for e in epochs]
        return epochs

    def get(self, epoch):
        model_path, metric_path = self.paths_from_epoch(epoch)
        if not model_path.exists() or not metric_path.exists():
            print("Cache doesn't contain given element")
            return None, None

        try:
            tensors = torch.load(model_path)
        except Exception:
            print("mapping on cpu")
            tensors = torch.load(model_path, map_location=torch.device('cpu'))
        
        data = read_json(metric_path)    
        return tensors, data
    
    def save(self, model, metrics, epoch):
        model_path, metric_path = self.paths_from_epoch(epoch)
        torch.save(model.state_dict(), model_path)
        metrics["epoch"] = epoch
        write_json(metrics, metric_path)
        self.latest_epoch = epoch
        
    def clear(self):
        shutil.rmtree(self.weights_path)
        shutil.rmtree(self.metric_path)
        self.weights_path.mkdir()
        self.metric_path.mkdir()

    def delete(self, epochs=[], metrics=False):
        for epoch in epochs:
            model_path, metric_path = self._paths(epoch)
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(metric_path) and metrics:
                os.remove(metric_path)
        
    def best_weights(self):
        all_epochs = self.get_epochs()
        best_params, best_iou, best_epoch = [], -1, -1
        for epoch in tqdm(all_epochs, total=len(all_epochs), leave=False, position=0):
            params, metrics = self.get(epoch)
            iou = metrics["mIoU"]
            if best_iou < iou:
                best_params = params
                best_iou = iou
                best_epoch = epoch
        print(f"best_epoch = {best_epoch}, best_iou = {best_iou}")
        return best_params