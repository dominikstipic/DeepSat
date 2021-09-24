from pathlib import Path
import argparse

from . import shared_logic as shared_logic
import pipeline.preprocess as preprocess
import src.utils.factory as factory
from src.transforms.transforms import Compose

FILE_NAME = Path(__file__).stem

def cmi_parse() -> dict:
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--input", default="data", help="Input directory")
    parser.add_argument("--output", default=f"{FILE_NAME}/output", help="Output directory")
    args = vars(parser.parse_args())
    args = {k: Path(v) for k,v in args.items()}
    config_path = args["config"]
    args["config"] = shared_logic.get_pipeline_stage_args(config_path, FILE_NAME)
    return config_path, args

def prepare_pip_arguments(config: dict, input: Path, output: Path):
    args = {}
    preprocs = []
    for preproc_dict in config["preprocess"]:
        preproc_fun = factory.import_object(preproc_dict)
        preprocs.append(preproc_fun)
    args["dataset"] = factory.get_object_from_standard_name(config["dataset"])(root=input, transforms=Compose(preprocs))
    args["format"] = config["format"]
    args["output_dir"] = output
    return args

if __name__ == "__main__":
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    preprocess.process(**processed_args)
