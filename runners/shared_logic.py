import sys
from pathlib import Path
import logging
import os
import traceback

import src.utils.compiler.config_compiler as config_compiler
import src.utils.compiler.actions as config_actions
from src.utils.common import read_json

JSON_PATH = Path("config.json")

def get_pipeline_stage_args(file_name):
    global JSON_PATH 
    config = read_json(JSON_PATH)
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


