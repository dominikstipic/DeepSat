"""
    Saves detections for the given dataset
"""
import importlib.util
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import src.transforms as TF
from src.model import DS_Model
from src.dataset.datasets import CamVid
from src.metrics import recall, precission, accuracy, mIoU
from src.metrics.observers import Confusion_Matrix
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
train_ld = DataLoader(train_db, batch_size=1, num_workers=workers, pin_memory=True, shuffle=True, drop_last=True)
valid_ld = DataLoader(valid_db, batch_size=1, num_workers=workers, pin_memory=True)
test_ld  = DataLoader(test_db, batch_size=1, num_workers=workers, pin_memory=True)
train_valid_ld = DataLoader(train_valid_db, batch_size=1, num_workers=workers, pin_memory=True)

###################### Learning
model = models.PiramidSwiftnet(num_classes=class_num)
model.loss_function = nn.CrossEntropyLoss(ignore_index=void_label, reduction="mean")

best = torch.load("/content/drive/MyDrive/DSLearn/resources/weights/piramida_best.zip")
model.load_state_dict(best)

loader = test_ld
target = Path(config["IMAGE_DIR"]) / "detections"
if target.exists(): shutil.rmtree(target)
target.mkdir()

model.eval()
with torch.no_grad():
    for i, (x,y) in tqdm(enumerate(loader), total=len(loader), leave=False, position=0):
        logits = model(x)
        y_pred = logits.argmax(1).squeeze().cpu().numpy()
        plt.figure(figsize=(12,12))
        plt.imshow(y_pred.squeeze())
        plt.axis("off")
        name = f"{i}.png"
        path = target / name
        plt.savefig(path)
        plt.close()




