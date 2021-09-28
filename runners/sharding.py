from pathlib import Path
import argparse

from . import shared_logic as shared_logic
import pipeline.sharding as sharding
import src.utils.pipeline_repository as pipeline_repository

FILE_NAME = Path(__file__).stem

def cmi_parse() -> dict:
    INPUT  = Path("preprocess/output")
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

def prepare_pip_arguments(config: dict, input: Path, output: Path):
    args = {}
    args["shard_size"] = config["shard_size"]
    args["input_dir"]  = input
    args["output_dir"] = output
    return args

def process():
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    sharding.process(**processed_args)

if __name__ == "__main__":
    process()
