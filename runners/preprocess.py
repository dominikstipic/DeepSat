from pathlib import Path
import argparse

from . import shared_logic as shared_logic
import pipeline.preprocess as preprocess
import src.utils.factory as factory
from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository


FILE_NAME = Path(__file__).stem

def cmi_parse() -> dict:
    INPUT  = "data"
    OUTPUT = Path(f"{FILE_NAME}/output")
    output_dir = pipeline_repository.get_path(OUTPUT)
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--input", default=INPUT, help="Input directory")
    parser.add_argument("--output", default=output_dir, help="Output directory")
    parser.add_argument("--in_alignment", dest="in_alignment", action="store_true", help="Monitor if input images and masks are aligned")
    parser.add_argument("--out_alignment", dest="out_alignment", action="store_true", help="Monitor if output images and masks are aligned")
    args = vars(parser.parse_args())
    
    in_alignment_value = args.pop("in_alignment")
    out_alignment_value = args.pop("out_alignment")
    args = {k: Path(v) for k,v in args.items()}
    args["in_alignment"] = in_alignment_value
    args["out_alignment"] = out_alignment_value

    config_path = args["config"]
    args["config"] = shared_logic.get_pipeline_stage_args(config_path, FILE_NAME)
    return config_path, args

def prepare_pip_arguments(config: dict, input: Path, output: Path, in_alignment: bool, out_alignment: bool):
    args = {}
    preprocs = []
    for preproc_dict in config["preprocess"]:
        preproc_fun = factory.import_object(preproc_dict)
        preprocs.append(preproc_fun)
    args["dataset"] = factory.get_object_from_standard_name(config["dataset"])(root=input, transforms=Compose(preprocs))
    args["format"] = config["format"]
    args["output_dir"] = output
    args["in_alignment"] = in_alignment
    args["out_alignment"] = out_alignment
    return args

def process():
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME, preprocess=True)
    preprocess.process(**processed_args)

if __name__ == "__main__":
    process()
