from pathlib import Path
import os

from . import shared_logic as shared_logic
import src.utils.factory as factory
import pipeline.data_stat as data_stat
import src.utils.pipeline_repository as piepline_repository

FILE_NAME = Path(__file__).stem
INPUT  = Path("dataset_factory/output")
OUTPUT = Path(f"{FILE_NAME}/output") 

def prepare_pip_arguments(config_args: dict):
    global INPUT, OUTPUT
    args = {}
    split_dirs = piepline_repository.get_path(INPUT)
    split_dict = piepline_repository.get_objects_from_repo(split_dirs)
    args.update(split_dict)
    args["viz_samples"] = config_args["viz_samples"]
    args["output"] = OUTPUT
    return args

if __name__ == "__main__":
    args = shared_logic.get_pipeline_stage_args(FILE_NAME)
    shared_logic.log_arguments(FILE_NAME, args)
    processed_args = prepare_pip_arguments(args)
    data_stat.process(**processed_args)
