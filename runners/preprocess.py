from pathlib import Path
import copy

from . import shared_logic as shared_logic
import pipeline.preprocess as preprocess
import src.utils.factory as factory
from src.transforms.transforms import Compose

INPUT = "data/AerialImageDataset"

def prepare_pip_arguments(config_args: dict):
    global INPUT
    args = {}
    preprocs = []
    for preproc_dict in config_args["preprocess"]:
        preproc_fun = factory.import_object(preproc_dict)
        preprocs.append(preproc_fun)
    args["dataset"] = factory.get_object_from_standard_name(config_args["dataset"])(root=INPUT, transforms=Compose(preprocs))
    args["format"] = config_args["format"]
    args["output_dir"] = Path(f"{Path(__file__).stem}/output")
    return args

if __name__ == "__main__":
    file_name = Path(__file__).stem
    args = shared_logic.get_pipeline_stage_args(file_name)
    shared_logic.log_arguments(file_name, args)
    processed_args = prepare_pip_arguments(args)
    preprocess.process(**processed_args)
