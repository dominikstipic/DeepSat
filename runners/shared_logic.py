import sys
from pathlib import Path
import logging
import os
import traceback

import src.utils.compiler.config_compiler as config_compiler
import src.utils.compiler.actions as config_actions
from src.utils.common import read_json
import src.utils.pipeline_repository as pipeline_repository

JSON_PATH = Path("config.json")

def get_pipeline_stage_args(file_name, compile=True):
    global JSON_PATH 
    config = read_json(JSON_PATH)
    if compile:
        try:
            actions = [config_actions.reference_action]
            config = config_compiler.compile(config, actions)
        except Exception:
            traceback.print_exc()
            print("Could not parse the configuration file!")
            sys.exit(0)
    pipeline_stage_args = config[file_name]
    return pipeline_stage_args

def log_arguments(file_name, args):
    logger = logging.getLogger(file_name)
    logger.info(f"{os.path.basename(file_name)} script was run with following arguments: {args}")

def save_args(stage_name: str):
    args = get_pipeline_stage_args(stage_name, compile=False)
    stage_name = Path(stage_name) 
    name = "runned_with.json"
    pipeline_repository.push_json(stage_name, name, args)
