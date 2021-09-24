from pathlib import Path
from runners.sharding import FILE_NAME
import argparse

import numpy as np

from . import shared_logic as shared_logic
from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import pipeline.dataset_factory as dataset_factory

FILE_NAME = Path(__file__).stem

def cmi_parse() -> dict:
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--input", default="sharding/output", help="Input directory")
    parser.add_argument("--output", default=f"{FILE_NAME}/output", help="Output directory")
    args = vars(parser.parse_args())
    args = {k: Path(v) for k,v in args.items()}
    config_path = args["config"]
    args["config"] = shared_logic.get_pipeline_stage_args(config_path, FILE_NAME)
    return config_path, args

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

def prepare_pip_arguments(config: dict, input: Path, output: Path) -> dict:
    viz_samples = config["viz_samples"]
    test_ratio, valid_ratio = config["test_ratio"], config["valid_ratio"]
    
    train_aug, test_aug = _get_train_test_transformations(config["augmentations"])
    train_tensor_tf, test_tensor_tf = _get_train_test_transformations(config["tensor_transf"])

    input_dir = pipeline_repository.get_path(input)
    dataset = factory.get_object_from_standard_name(config["dataset"])(input_dir, [])
    d = dict(viz_samples=viz_samples, test_ratio=test_ratio, valid_ratio=valid_ratio, 
             train_aug=train_aug, test_aug=test_aug, train_tensor_tf=train_tensor_tf, test_tensor_tf=test_tensor_tf, 
             input_dir=input_dir, dataset=dataset)
    return d

if __name__ == "__main__":
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    dataset_factory.process(**processed_args)