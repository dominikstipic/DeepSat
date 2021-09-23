from pathlib import Path

from . import shared_logic as shared_logic
import pipeline.preprocess as preprocess
import src.utils.factory as factory
from src.transforms.transforms import Compose

FILE_NAME = Path(__file__).stem
INPUT  = Path("data")
OUTPUT = Path(f"{FILE_NAME}/output") 

def prepare_pip_arguments(config_args: dict):
    global INPUT, OUTPUT
    args = {}
    preprocs = []
    for preproc_dict in config_args["preprocess"]:
        preproc_fun = factory.import_object(preproc_dict)
        preprocs.append(preproc_fun)
    args["dataset"] = factory.get_object_from_standard_name(config_args["dataset"])(root=INPUT, transforms=Compose(preprocs))
    args["format"] = config_args["format"]
    args["output_dir"] = OUTPUT
    return args

if __name__ == "__main__":
    args = shared_logic.prerun_routine(FILE_NAME)
    processed_args = prepare_pip_arguments(args)
    preprocess.process(**processed_args)
