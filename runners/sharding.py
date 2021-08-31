from pathlib import Path
import os

from . import shared_logic as shared_logic

if __name__ == "__main__":
    file_name = Path(__file__).stem
    args = shared_logic.get_pipeline_stage_args(file_name)
    shard_size = args["shard_size"]
    shared_logic.log_arguments(file_name, args)
    os.system(f"python -m pipeline.{file_name} {shard_size}")
