"""
    Loads best weights from the storage and creates the report
"""

import importlib.util

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import src.transforms as TF
from src.model import DS_Model
from src.dataset.datasets import CamVid
from src.utils.report import report
import src.model.models as models

def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

env = import_module("src/init_script.py")
config = env.config
hiperparams = config["HIPER_PARAMS"]

######################## Hiperparameters
batch_size = hiperparams["BATCH_SIZE"]
void_label = -1
class_num = 11

######################## Transformations
normalization = {"mean" : CamVid.mean, 
                 "std" :  CamVid.std}
train_tf = TF.Compose([TF.Flipper(0.5),
                      TF.RandomSquareCropAndScale(448),
                      TF.Tensor(normalization)])
valid_tf = TF.Compose([TF.Tensor(normalization)])

######################## Dataset
root = config["DATASET"]
train_db = CamVid(root, "train", train_tf)
valid_db = CamVid(root, "val", valid_tf)
test_db  = CamVid(root, "test", valid_tf)
train_valid_db = CamVid.combine([train_db, valid_db], train_tf)

workers = 4
train_ld = DataLoader(train_db, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True, drop_last=True)
valid_ld = DataLoader(valid_db, batch_size=1, num_workers=workers, pin_memory=True)
test_ld  = DataLoader(test_db, batch_size=1, num_workers=workers, pin_memory=True)
train_valid_ld = DataLoader(train_valid_db, batch_size=batch_size, num_workers=workers, pin_memory=True)

###################### Learning
model = models.PiramidSwiftnet(num_classes=class_num)
loss_function = nn.CrossEntropyLoss(ignore_index=void_label, reduction="mean")
model.loss_function = loss_function

#storage = env.Storage.get()
#best = storage.best_weights()
weights = torch.load("/content/drive/MyDrive/DSLearn/resources/weights/piramida_best.zip")
model.load_state_dict(weights)

#model.fit(epochs=hiperparams["EPOCHS"], device="cuda")
report(model, test_db, config, N=6)
