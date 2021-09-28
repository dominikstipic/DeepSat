import os
from pathlib import Path
from pdb import run
from src.utils import common, hashes
import argparse
import time
import sys
import importlib
import traceback
import shutil

import torch
import yagmail

import src.utils.pipeline_repository as pipeline_repository
import devops.commit as commit

_EMAIL_PATH  = Path("email.json")
REPOSITORY_PATH = pipeline_repository.get_path("")
REPORT_PATH = Path("reports")
META_JSON = "runned_with.json"
#####################################

def cmi_parse() -> tuple:
    parser = argparse.ArgumentParser(description="DeepSat pipeline executor")
    parser.add_argument("--do_report", dest="do_report", action="store_true", help="Generate pipeline report and save it to a report repository")
    parser.add_argument("--do_email", dest="do_email", action="store_true", help="Send the report to the specified receiver email")
    parser.add_argument("--do_version", dest="do_version", action="store_true", help="Version reports with dvc")
    parser.add_argument("--config_path", default="config.json", type=str, help="Path to the configuration path")
    parser.add_argument("--data_path", default="data", type=str, help="Path to the dataset directory")
    args = vars(parser.parse_args())
    return args

def send_email(config: dict, eval: dict, time: int, **kwargs):
    email_dict = common.read_json(_EMAIL_PATH)
    yagmail.register(email_dict["email"], email_dict["pass"])
    results = eval["metrics"]
    results["time"] = f"{time} min" 
    contents = [
        results,
        "###########################################", 
        config
    ]
    yagmail.SMTP(email_dict["email"]).send(email_dict["receiver"], email_dict["subject"], contents)

def version_report():
    commit_type = "run" 
    test = False
    message = ""
    commit.process(commit_type, message, test)

def get_config(config_path: str):
    config_dict = common.read_json(Path(config_path), ordered_dict=True)
    data_hash = hashes.current_data_hash()
    if "preprocess" in config_dict.keys():
        config_dict["preprocess"]["data_hash"] = data_hash
    return config_dict

def generate_report(config_dict: dict):
    time = common.current_time()
    root_dir = REPORT_PATH / time
    if not root_dir.exists(): 
        if not root_dir.parent.exists(): root_dir.parent.mkdir()
        root_dir.mkdir()
    trainer_path = Path("trainer/output")
    trainer_out = pipeline_repository.get_objects(trainer_path)
    weights, model = trainer_out["weights"], trainer_out["model"]
    eval_path = Path("evaluation/artifacts")
    eval_out = pipeline_repository.get_objects(eval_path)["metrics"]

    torch.save(weights, str(root_dir / 'weights.pt'))
    common.save_pickle(str(root_dir / "model.pickle"), model)
    common.write_json(eval_out, root_dir / "eval.json")
    common.write_json(config_dict, root_dir / "config.json")
    
    report = dict(weights=weights, model=model, eval=eval_out, config=config_dict)
    return report

def run_stage(stage, config_path):
    run_cmd = f"python -m runners.{stage} --config={config_path}"
    os.system(run_cmd)

def process(do_report: bool, do_version: bool, do_email: bool, config_path: str, data_path: str):
    config = get_config(config_path)
    pipeline_stages = config.keys()
    flag = True
    start = time.perf_counter_ns()
    to_min = lambda t : t / 1000 / 1000 / 60
    if not Path(data_path).exists() or len(list(Path(data_path).iterdir())) == 0:
        raise RuntimeError("cannot find dataset on which system will learn")
    for stage_name in pipeline_stages:
        pip_stage_path = pipeline_repository.get_path(stage_name)
        runned_with_path = pip_stage_path / META_JSON
        runned_with = common.read_json(runned_with_path)
        if runned_with == None or config[stage_name] != runned_with or not flag:
            print(f"RUNNING: {stage_name}")
            flag = False
            try:
                run_stage(stage_name, config_path)
            except Exception:
                shutil.rmtree(f"repository/{stage_name}")
                print(f"stage failed: {stage_name}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print(f"SKIPPING: {stage_name}")
    end = time.perf_counter_ns()
    time_min = to_min(end-start)
    print(f"TOTAL TIME: {time_min}")
    if do_report:
        report = generate_report(config)
        report["time"] = time_min
        if do_email: send_email(**report)
        if do_version: version_report()

if __name__ == "__main__":
    args = cmi_parse()
    print(args)
    process(**args)