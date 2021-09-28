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
    INPUT  = Path("sharding/output")
    OUTPUT = Path(f"{FILE_NAME}/output")
    input_dir, output_dir  = pipeline_repository.get_path(INPUT), pipeline_repository.get_path(OUTPUT)
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--input", default=input_dir, help="Input directory")
    parser.add_argument("--output", default=output_dir, help="Output directory")
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
    result = {}
    for split_name, tf_dict in trainsformation_dict.items():
        transf_composite =_get_composite_transf(tf_dict)
        result[split_name] = transf_composite
    return result

def prepare_pip_arguments(config: dict, input: Path, output: Path) -> dict:
    viz_samples = config["viz_samples"]
    test_ratio, valid_ratio = config["test_ratio"], config["valid_ratio"]
    
    aug_dict    = _get_train_test_transformations(config["augmentations"])
    tensor_dict = _get_train_test_transformations(config["tensor_transf"])

    dataset = factory.get_object_from_standard_name(config["dataset"])(input, [])
    d = dict(viz_samples=viz_samples, test_ratio=test_ratio, valid_ratio=valid_ratio, 
             aug_dict=aug_dict, tensor_tf_dict=tensor_dict, input_dir=input, dataset=dataset)
    return d

def process():
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    dataset_factory.process(**processed_args)

if __name__ == "__main__":
    process()