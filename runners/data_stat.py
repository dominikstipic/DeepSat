from pathlib import Path
import argparse
from src.utils import common

from . import shared_logic as shared_logic
import pipeline.data_stat as data_stat
import src.utils.pipeline_repository as piepline_repository

FILE_NAME = Path(__file__).stem

def cmi_parse() -> dict:
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--input", default="dataset_factory/output", help="Input directory")
    parser.add_argument("--output", default=f"{FILE_NAME}/output", help="Output directory")
    args = vars(parser.parse_args())
    args = {k: Path(v) for k,v in args.items()}
    config_path = args["config"]
    args["config"] = shared_logic.get_pipeline_stage_args(config_path, FILE_NAME)
    return config_path, args

def prepare_pip_arguments(config: dict, input: Path, output: Path):
    args = {}
    input, output = Path(input), Path(output)
    split_dirs = piepline_repository.get_path(input)
    split_dict = piepline_repository.get_objects(split_dirs)
    args.update(split_dict)
    args["viz_samples"] = config["viz_samples"]
    args["output"] = output
    return args

if __name__ == "__main__":
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    data_stat.process(**processed_args)
