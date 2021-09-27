from pathlib import Path
import copy
import argparse

from . import shared_logic as shared_logic
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import src.observers.metrics as metrics
import pipeline.trainer as trainer

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

def _get_datasets(input_path: Path):
    previous_stage_obj_dict = pipeline_repository.get_objects(input_path)
    train_db, valid_db = previous_stage_obj_dict["train_db"], previous_stage_obj_dict["valid_db"]
    return dict(train_db=train_db, valid_db=valid_db)

def _create_dataloaders(dataloader_dict: dict, train_db, valid_db):
    train_params, valid_params = dataloader_dict["train"], dataloader_dict["valid"]
    train_dl = factory.import_object(train_params, train_db=train_db)
    valid_dl = factory.import_object(valid_params, valid_db=valid_db)
    return dict(train_dl=train_dl, valid_dl=valid_dl)

def get_observers(observer_dict: dict): 
    results = {key:[] for key in observer_dict}
    for event_key, event_obs in copy.deepcopy(observer_dict).items():
        for obs_dict in event_obs:
            metrics_funs = metrics.METRIC_FUNS
            obs = factory.import_object(obs_dict, **metrics_funs)
            results[event_key].append(obs)
    return results

def prepare_pip_arguments(config: dict, input: Path, output: Path):
    args = {} 
    args["model"] = factory.import_object(config['model'])
    datasets_dict = _get_datasets(input)
    args["loader_dict"] = _create_dataloaders(config["dataloader"], **datasets_dict)
    args["optimizer"] = factory.import_object(config["optimizer"], model=args["model"])
    args["loss_function"] = factory.import_object(config['loss_function'])
    args["lr_scheduler"] = factory.import_object(config['lr_scheduler'], optimizer=args["optimizer"])
    args["observers_dict"] = get_observers(config["observers"])
    args["epochs"] = config["epochs"]
    args["device"] = config["device"]
    args["amp"] = config["amp"]
    args["output_dir"] = output
    return args

if __name__ == "__main__":
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    trainer.process(**processed_args)

