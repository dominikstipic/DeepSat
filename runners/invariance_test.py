from os import pipe
from pathlib import Path
import copy
import argparse

from . import shared_logic as shared_logic
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import pipeline.invariance_test as invariance_test
from runners.evaluation import get_model
import runners.trainer as trainer


FILE_NAME = Path(__file__).stem

def cmi_parse() -> dict:
    MODEL_INPUT   = Path("trainer/output")
    DATASET_INPUT  = Path("dataset_factory/output/test_db.pickle")
    OUTPUT = Path(f"{FILE_NAME}/output")
    model_input = pipeline_repository.get_path(MODEL_INPUT)
    dataset_input = pipeline_repository.get_path(DATASET_INPUT)
    output_dir = pipeline_repository.get_path(OUTPUT)
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--model_input", default=model_input, help="Input directory")
    parser.add_argument("--dataset_input", default=dataset_input, help="Input directory")
    parser.add_argument("--output", default=output_dir, help="Output directory")
    args = vars(parser.parse_args())
    args = {k: Path(v) for k,v in args.items()}
    config_path = args["config"]
    args["config"] = shared_logic.get_pipeline_stage_args(config_path, FILE_NAME)
    return config_path, args

def get_dataset(dataset_path: Path):
    if dataset_path.name.endswith(".pickle"):
        dataset = pipeline_repository.get_pickle(dataset_path)
    else:
        ## ako su tarovi u direktoriju onda na TarDataset
        ## Inace Inria Dataset
        raise NotImplemented("not implemented for directories!")
    return dataset

def prepare_pip_arguments(config: dict, dataset_input: Path, model_input: Path, output: Path):
    args = {}
    attributes = config["attributes"]
    attributes_tf = [factory.import_object(attribute_dict) for attribute_dict in attributes]
    args["attributes"] = list(zip(attributes, attributes_tf))
    args["dataset"] = get_dataset(dataset_input)
    args["model"]   = get_model(model_input)
    args["output_dir"] = output
    args["device"] = config["device"]
    args["observers_dict"] = trainer.get_observers(config["observers"])
    return args

def process():
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    invariance_test.process(**processed_args)

if __name__ == "__main__":
    process()

