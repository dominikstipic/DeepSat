from pathlib import Path 
from functools import partial

import numpy as np
import torch
import torchvision.transforms as transforms
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from src.models.piramid_swiftnet.model import PiramidSwiftnet
from src.datasets.tar_dataset import TarDataset
import src.transforms.transforms as transforms
import src.observers.subscribers as subscribers 
import src.observers.metrics as metrics 

def get_loaders(datasets: dict, bs: int):
  train_db, test_db = datasets["train"], datasets["test"]
  train_dl = torch.utils.data.DataLoader(train_db, batch_size=bs)
  test_dl  = torch.utils.data.DataLoader(test_db, batch_size=bs)
  return train_dl, test_dl

def get_model(config):
  model = PiramidSwiftnet(2)
  params = [
            {"params": model.random_init_params(), 
             "lr": config["lr1"],
             "weight_decay": config["wd1"]},
             {"params": model.fine_tune_params(), 
             "lr": config["lr2"],
             "weight_decay": config["wd2"]}
          ]
  model.optimizer = torch.optim.Adam(params=params)
  model.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
  cm = subscribers.Confusion_Matrix(class_num=2, metrics=[metrics.mIoU])
  model.observers={"after_epoch": [], "after_step": [cm], "before_step": [], "before_epoch": []}
  model.device = "cpu"
  return model

def get_datasets(path: Path):
  train_path, test_path = path / "train", path / "test"
  tf = transforms.To_Tensor(mean=[103.2342, 108.9520, 100.1419], 
                            std=[48.6281, 44.4967, 41.9143], 
                            input_type=np.float32, 
                            label_type=np.int64)
  train_db = TarDataset(train_path, tf)
  test_db  = TarDataset(test_path, tf)
  return dict(train=train_db, test=test_db)

def training_function(config, datasets, batch_size):
    model = get_model(config)
    train_dl, test_dl = get_loaders(datasets, batch_size)
    model.train_loader = train_dl
    model.valid_loader = test_dl
    model.train_state()
    for _ in range(1):
      model.one_epoch()
      model.evaluate()
      results = model.observer_results()
      mIoU = results["mIoU"]
      tune.report(performance=mIoU)

####################

search_space = {
    "lr1": tune.uniform(2e-4, 6e-4),
    "lr2": tune.uniform(1e-4, 4e-4),
    "wd1": tune.uniform(1e-4, 4e-4),
    "wd2": tune.uniform(1e-4, 4e-4)
}
search_alg = BasicVariantGenerator()
path = Path("experiments/hypertuner/data")
datasets = get_datasets(path)
batch_size = 16

analysis = tune.run(
  partial(training_function, datasets=datasets, batch_size=batch_size),
  metric="performance",
  mode="max",
  config=search_space, 
  search_alg=search_alg,
  resources_per_trial={"cpu": 1},
  num_samples=1,
  verbose=0, 
  checkpoint_at_end=False)

df = analysis.results_df
df.to_csv("data.csv")

