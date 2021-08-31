from pathlib import Path
import copy
import os
import numpy as np

from . import shared_logic as shared_logic
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import src.observers.metrics as metrics
import pipeline.trainer as trainer

from src.transforms.transforms import To_Tensor
from src.datasets.inria import Inria
import torch

def get_datasets():
    previous_stage_obj_dict = pipeline_repository.get_objects_from_repo("dataset_factory", Path("output/datasets"))
    train_db, valid_db, test_db = previous_stage_obj_dict["train_db"], previous_stage_obj_dict["valid_db"], previous_stage_obj_dict["test_db"]

    ###### TODO: HACK
    tf = To_Tensor(Inria.mean, Inria.std, np.float32, np.int64)
    train_db.transform.transforms.append(tf)
    valid_db.transform.transforms.append(tf)
    test_db.transform.transforms.append(tf)
    ###### TODO: HACK
    
    return dict(train_db=train_db, valid_db=valid_db, test_db=test_db)

def create_dataloaders(args, train_db, valid_db, test_db):
    dataloader_dict = args["dataloader"]
    train_params, valid_params, test_params = dataloader_dict["train"], dataloader_dict["valid"], dataloader_dict["test"]
    train_dl = factory.import_object(train_params, train_db=train_db)
    valid_dl = factory.import_object(valid_params, valid_db=valid_db)
    test_dl = factory.import_object(test_params, test_db=test_db)
    return dict(train_dl=train_dl, valid_dl=valid_dl, test_dl=test_dl)

def get_observers(observer_dict: dict): 
    results = {key:[] for key in observer_dict}
    for event_key, event_obs in copy.deepcopy(observer_dict).items():
        for obs_name, obs_params in event_obs.items():
            metrics_funs = metrics.METRIC_FUNS
            obs = factory.import_object({obs_name:obs_params}, **metrics_funs)
            results[event_key].append(obs)
    return results

def prepare_dicitionary(config_args: dict):
    args = {} 
    args["model"] = factory.import_object(config_args['model'])
    datasets_dict = get_datasets()
    args["loader_dict"] = create_dataloaders(config_args, **datasets_dict)
    args["optimizer"] = factory.import_object(config_args["optimizer"], model=args["model"])
    args["loss_function"] = factory.import_object(config_args['loss_function'])
    args["lr_scheduler"] = factory.import_object(config_args['lr_scheduler'], optimizer=args["optimizer"])
    args["observers_dict"] = get_observers(config_args["observers"])
    args["epochs"] = config_args["epochs"]
    args["device"] = config_args["device"]
    args["amp"] = config_args["amp"]
    return args

if __name__ == "__main__":
    file_name = Path(__file__).stem
    args = shared_logic.get_pipeline_stage_args(file_name)
    shared_logic.log_arguments(file_name, args)
    processed_args = prepare_dicitionary(args)
    trainer.process(**processed_args)

