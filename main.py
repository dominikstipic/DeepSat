import os
from pathlib import Path
from src.utils import common, hashes
import argparse
import time
import sys
import traceback
import shutil

import torch
import yagmail

import src.utils.pipeline_repository as pipeline_repository
import scripts.commit as commit

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
    parser.add_argument("--force_eval", dest="force_eval", action="store_true", help="Force pipeline run")
    args = vars(parser.parse_args())
    return args

def send_email(time: int, config: dict):
    metric_path = pipeline_repository.get_path("evaluation/output")
    results = pipeline_repository.get_objects(metric_path)["metrics"]
    results["time"] = f"{time} min"
    email_dict = common.read_json(_EMAIL_PATH)
    yagmail.register(email_dict["email"], email_dict["pass"])
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

def run(stage_name: str, config_path: Path):
    print(f"running: {stage_name}")
    try:
        run_cmd = f"python -m runners.{stage_name} --config={config_path}"
        os.system(run_cmd)
    except Exception:
        shutil.rmtree(f"repository/{stage_name}")
        print(f"stage failed: {stage_name}")
        traceback.print_exc()
        sys.exit(1)

def run_pipeline_stage(stage_name: str, config_path: Path, previous_params: dict, current_params: dict, previous_phase_runned: bool, force_eval: bool):
    can_skip_stage = lambda previous_params, current_params: previous_params != None and current_params == previous_params
    print(stage_name)
    if force_eval or not can_skip_stage(previous_params, current_params) or previous_phase_runned:
        stage_path = Path(f"repository/{stage_name}")
        if stage_path.exists(): shutil.rmtree(str(stage_path))
        run(stage_name, config_path)
        return True
    print(f"skipping: {stage_name}")
    return False
    
def process(do_report: bool, do_version: bool, do_email: bool, force_eval: bool, config_path: str, data_path: str):
    config = get_config(config_path)
    pipeline_stages = config.keys()
    start = time.perf_counter_ns()
    to_min = lambda t : t / (1e9 * 60)
    print(f"DATA PATH: {data_path}")
    if not Path(data_path).exists() or len(list(Path(data_path).iterdir())) == 0:
        raise RuntimeError("cannot find dataset on which system will learn")
    previous_phase_runned = False
    for stage_name in pipeline_stages:
        pip_stage_path = pipeline_repository.get_path(stage_name)
        runned_with_path = pip_stage_path / META_JSON
        current_params, previous_params = config[stage_name], common.read_json(runned_with_path)
        pipeline_runned = run_pipeline_stage(stage_name, config_path, previous_params, current_params, previous_phase_runned, force_eval)
        previous_phase_runned = previous_phase_runned or pipeline_runned
    end = time.perf_counter_ns()
    time_min = to_min(end-start)
    print(f"TOTAL TIME: {time_min}")
    if do_email:
        send_email(time_min, config)

if __name__ == "__main__":
    args = cmi_parse()
    print(args)
    process(**args)