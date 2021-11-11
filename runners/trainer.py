from pathlib import Path
import copy
import argparse

import ray

from . import shared_logic as shared_logic
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import src.observers.metrics as metrics
import pipeline.trainer as trainer
import src.hypertuner as hypertuner
import src.utils.compiler.actions as actions

FILE_NAME = Path(__file__).stem

def cmi_parse() -> dict:
    INPUT  = Path("dataset_factory/output")
    OUTPUT = Path(f"{FILE_NAME}/output")
    input_dir, output_dir = pipeline_repository.get_path(INPUT), pipeline_repository.get_path(OUTPUT)
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--input", default=input_dir, help="Input directory")
    parser.add_argument("--output", default=output_dir, help="Output directory")
    args = vars(parser.parse_args())
    args = {k: Path(v) for k,v in args.items()}
    config_path = args["config"]
    args["config"] = shared_logic.get_pipeline_stage_args(config_path, FILE_NAME)
    return config_path, args

def _create_dataloaders(dataloader_dict: dict, datasets: dict):
    dataloaders = {}
    for split, dataloader_params in dataloader_dict.items():
        dl = factory.import_object(dataloader_params, **datasets)
        dataloaders[split] = dl
    return dataloaders

def get_observers(observer_dict: dict): 
    results = {key:[] for key in observer_dict}
    for event_key, event_obs in copy.deepcopy(observer_dict).items():
        for obs_dict in event_obs:
            metrics_funs = metrics.METRIC_FUNS
            obs = factory.import_object(obs_dict, **metrics_funs)
            results[event_key].append(obs)
    return results

##################

def get_search_algorithm(search_alg_string: str):
    if search_alg_string not in hypertuner.search_algs.keys():
        raise RuntimeError("Search algorithm not available")
    return hypertuner.search_algs[search_alg_string]

def get_hypertuner(hypertuning_dict: dict):
    eval_action = actions.eval_action_init({"ray":ray})
    hs = hypertuning_dict["search_space"]
    search_space = {k: eval_action(v) for k,v in hs.items()}
    search_algorithm = get_search_algorithm(hypertuning_dict["search_alg"])
    num_samples = hypertuning_dict["num_samples"]
    resources_per_trial = hypertuning_dict["resources_per_trial"]
    iters = hypertuning_dict["iterations"]
    tuner = hypertuner.HyperTuner(search_space, search_algorithm, resources_per_trial, num_samples, iters)
    tuner.active = hypertuning_dict["active"]
    return tuner

##################

def prepare_pip_arguments(config: dict, input: Path, output: Path):
    args = {} 
    args["model"] = factory.import_object(config['model'])
    datasets_dict = pipeline_repository.get_objects(input)
    args["loader_dict"] = _create_dataloaders(config["dataloader"], datasets_dict)
    args["optimizer"] = factory.import_object(config["optimizer"], model=args["model"])
    args["loss_function"] = factory.import_object(config['loss_function'])
    args["lr_scheduler"] = factory.import_object(config['lr_scheduler'], optimizer=args["optimizer"])
    args["observers_dict"] = get_observers(config["observers"])
    args["epochs"] = config["epochs"]
    args["device"] = config["device"]
    args["amp"] = config["amp"]
    args["mixup_factor"] = config["mixup_factor"]
    args["output_dir"] = output
    args["hypertuner"] = get_hypertuner(config["hypertuner"])
    args["active"] = config["active"]
    return args

def process():
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    trainer.process(**processed_args)

if __name__ == "__main__":
    process()

