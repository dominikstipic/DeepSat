from src.utils import hashes
import sys
from pathlib import Path
import logging
import os
import traceback
import argparse

import src.utils.compiler.config_compiler as config_compiler
import src.utils.compiler.actions as config_actions
from src.utils.common import read_json
import src.utils.pipeline_repository as pipeline_repository
import src.utils.hashes as hashes

def prerun_routine(file_name):
    config = Path(cmi_parse().config)
    args = get_pipeline_stage_args(config, file_name)
    save_args(config, file_name)
    log_arguments(file_name, args)
    return args

def cmi_parse() -> tuple:
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    args = parser.parse_args()
    return args

def get_pipeline_stage_args(config_path: Path, file_name: str, compile=True):
    config = read_json(config_path)
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

def save_args(config_path: Path, stage_name: str, preprocess=False):
    args = get_pipeline_stage_args(config_path, stage_name, compile=False)
    if preprocess:
        args["data_hash"] = hashes.current_data_hash()
    stage_name = Path(stage_name) 
    name = "runned_with.json"
    pipeline_repository.push_json(stage_name, name, args)
