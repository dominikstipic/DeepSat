from pathlib import Path
from copy import deepcopy
from itertools import chain

import torch
import numpy as np
from tqdm import tqdm

class Sat_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms=None):
        self.dataset_root = Path(root)
        self.split = split
        self.transforms = transforms
        self._data   = None
        self._labels = None

    def __len__(self):
         return len(self.data)
        
    @property
    def data(self):
        return self._data
    
    @property
    def labels(self):
        return self._labels
    
    @data.setter
    def data(self, data):
        if type(data) == list:
            self._data = np.array(data)
        else:
            self._data = data
    
    @labels.setter
    def labels(self, labels):
        self._labels = np.array(labels)

    def sample(self, n):
        sampled = deepcopy(self)
        idx = np.random.choice(len(sampled), size=n, replace=False)
        sampled.data   = np.array(self.data)[idx]
        sampled.labels = np.array(self.labels)[idx]
        return sampled
    
    def summary(self):
        mean = torch.zeros(3, dtype=torch.float)
        std  = torch.zeros(3, dtype=torch.float)
        N = len(self)
        c,h,w = None, None, None
        for i in tqdm(range(N)):
            x,_ = self[i]
            c,h,w = x.shape
            x = x.reshape(c,-1)
            mean += x.mean(1)
            std += x.std(1)
        mean /= N
        std /= N
        d = dict(length=N, mean=mean.numpy(), std=std.numpy(), channels=c, height=h, width=w)
        return d

    
    @staticmethod
    def combine(datasets, tf=None):
        assert len(datasets) > 0, "dataset is empty!"
        transforms = [d.transforms for d in datasets]
        if not tf:
            tf = datasets[0].transforms
            print("Transformation not given. Combined dataset will have transformation taken from the first dataset in the list")
        a = datasets[0]
        a.transforms = tf
        datasets_data = [d.data for d in datasets]
        datasets_labels = [d.labels for d in datasets]
        combined_data = list(chain(*datasets_data))
        combined_labels = list(chain(*datasets_labels))
        a.data   = np.array(combined_data)
        a.labels = np.array(combined_labels)
        return a

    def split_dataset(self, ratio):
        split1, split2 = deepcopy(self), deepcopy(self)
        n = len(self)
        split1_size = int(n*ratio)
        all_idx = np.arange(n)
        idx1 = np.random.choice(all_idx, size=split1_size, replace=False)
        idx2 = np.array(list(set(all_idx).difference(idx1)))
        split1.data, split2.data = self.data[idx1], self.data[idx2]
        split1.labels, split2.labels = self.labels[idx1], self.labels[idx2]
        return split1, split2
    
    def t(self, img, mask):
        if not self.transforms:
            img, mask = np.array(img, dtype=np.float32), np.array(mask, dtype=np.int64)
            img = np.transpose(img, (2,1,0))
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
            img, mask = torch.from_numpy(img), torch.from_numpy(mask)
            return img, mask
        else:
            img, mask = self.transforms([img,mask])
            return img, mask
    
    def __getitem__(self, idx):
        img, mask = self.get(idx)
        img, mask = self.t(img, mask)
        return img, mask

    def get(self, idx):
        pass

