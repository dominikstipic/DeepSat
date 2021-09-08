from pathlib import Path

from . import shared_logic as shared_logic
import pipeline.sharding as sharding

FILE_NAME = Path(__file__).stem
INPUT  = Path("preprocess/output")
OUTPUT = Path(f"{FILE_NAME}/output") 

def prepare_pip_arguments(config_args: dict):
    global INPUT, OUTPUT
    args = {}
    args["shard_size"] = config_args["shard_size"]
    args["input_dir"]  = INPUT
    args["output_dir"] = OUTPUT
    return args

if __name__ == "__main__":
    args = shared_logic.get_pipeline_stage_args(FILE_NAME)
    shared_logic.save_args(FILE_NAME)
    shared_logic.log_arguments(FILE_NAME, args)
    processed_args = prepare_pip_arguments(args)
    sharding.process(**processed_args)
