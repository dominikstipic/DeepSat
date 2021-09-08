import os
from pathlib import Path
from src.utils import common

import torch

import src.utils.pipeline_repository as pipeline_repository

PIPELINE = [
    "preprocess",
    "sharding",
    "dataset_factory",
    "data_stat",
    "trainer",
    "evaluation"
]

_CONFIG_PATH = Path("config.json")

REPOSITORY_PATH = pipeline_repository.get_path("")
REPORT_PATH = Path("reports")

META_JSON = "runned_with.json"

def get_config():
    config_dict = common.read_json(_CONFIG_PATH)
    return config_dict


def generate_report():
    time = common.current_time()
    root_dir = REPORT_PATH / time
    if not root_dir.exists(): root_dir.mkdir()

    trainer_path = Path("trainer/output")
    trainer_out = pipeline_repository.get_objects(trainer_path)
    weights, model = trainer_out["weights"], trainer_out["model"]
    eval_path = Path("evaluation/artifacts")
    eval_out = pipeline_repository.get_objects(eval_path)
    config_dict = get_config()

    torch.save(weights, str(root_dir / 'weights.pt'))
    common.save_pickle(str(root_dir / "model.pickle"), model)
    common.write_json(eval_out, root_dir / "eval.json")
    common.write_json(config_dict, root_dir / "config.json")

def run_stage(stage):
    get_cmd = lambda stage: f"python -m runners.{stage}" 
    cmd = get_cmd(stage)
    os.system(cmd)

def process(pipeline_stages: list):
    config = get_config()
    flag = True
    for stage_name in pipeline_stages:
        pip_stage_path = pipeline_repository.get_path(stage_name)
        runned_with_path = pip_stage_path / META_JSON
        runned_with = common.read_json(runned_with_path)
        if runned_with == None or config[stage_name] != runned_with or not flag:
            print(f"RUNNING: {stage_name}")
            flag = False
            run_stage(stage_name)
        else:
            print(f"SKIPPING: {stage_name}")
    generate_report()

if __name__ == "__main__":
    process(PIPELINE)