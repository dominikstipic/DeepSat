from pathlib import Path
from runners.sharding import FILE_NAME

import numpy as np

from . import shared_logic as shared_logic
from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import pipeline.dataset_factory as dataset_factory

FILE_NAME = Path(__file__).stem
INPUT  = Path("sharding/output") # TODO: What if I change sharding name

def _get_composite_transf(transformations: list) -> Compose: 
    transf_list = []
    for transf_dict in transformations: 
        transf = factory.import_object(transf_dict, np=np)
        transf_list.append(transf)
    return Compose(transf_list)

def _get_train_test_transformations(trainsformation_dict: dict) -> tuple:
     train_transf, test_transf = trainsformation_dict["train"], trainsformation_dict["test"]
     train_transf, test_transf = _get_composite_transf(train_transf), _get_composite_transf(test_transf)
     return train_transf, test_transf

def prepare_pip_arguments(config_args: dict) -> dict:
    global INPUT
    viz_samples = config_args["viz_samples"]
    test_ratio, valid_ratio = config_args["test_ratio"], config_args["valid_ratio"]
    
    train_aug, test_aug = _get_train_test_transformations(config_args["augmentations"])
    train_tensor_tf, test_tensor_tf = _get_train_test_transformations(config_args["tensor_transf"])

    input_dir = pipeline_repository.get_path(INPUT)
    dataset = factory.get_object_from_standard_name(config_args["dataset"])(input_dir, [])
    d = dict(viz_samples=viz_samples, test_ratio=test_ratio, valid_ratio=valid_ratio, 
             train_aug=train_aug, test_aug=test_aug, train_tensor_tf=train_tensor_tf, test_tensor_tf=test_tensor_tf, 
             input_dir=input_dir, dataset=dataset)
    return d

if __name__ == "__main__":
    args = shared_logic.get_pipeline_stage_args(FILE_NAME)
    shared_logic.log_arguments(FILE_NAME, args)
    processed_args = prepare_pip_arguments(args)
    dataset_factory.process(**processed_args)