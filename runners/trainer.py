from pathlib import Path
import copy

from . import shared_logic as shared_logic
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import src.observers.metrics as metrics
import pipeline.trainer as trainer

FILE_NAME = Path(__file__).stem
INPUT  = Path("dataset_factory/output") # TODO
OUTPUT = Path(f"{FILE_NAME}/output") 

def _get_datasets(input_path: Path):
    stage_name = input_path.parts[0]
    repo_path  = Path("/".join(input_path.parts[1:]))
    previous_stage_obj_dict = pipeline_repository.get_objects_from_repo(stage_name, repo_path)
    train_db, valid_db = previous_stage_obj_dict["train_db"], previous_stage_obj_dict["valid_db"]
    return dict(train_db=train_db, valid_db=valid_db)

def _create_dataloaders(dataloader_dict: dict, train_db, valid_db):
    train_params, valid_params = dataloader_dict["train"], dataloader_dict["valid"]
    train_dl = factory.import_object(train_params, train_db=train_db)
    valid_dl = factory.import_object(valid_params, valid_db=valid_db)
    return dict(train_dl=train_dl, valid_dl=valid_dl)

def _get_observers(observer_dict: dict): 
    results = {key:[] for key in observer_dict}
    for event_key, event_obs in copy.deepcopy(observer_dict).items():
        for obs_name, obs_params in event_obs.items():
            metrics_funs = metrics.METRIC_FUNS
            obs = factory.import_object({obs_name:obs_params}, **metrics_funs)
            results[event_key].append(obs)
    return results

def prepare_pip_arguments(config_args: dict):
    global INPUT, OUTPUT
    args = {} 
    args["model"] = factory.import_object(config_args['model'])
    datasets_dict = _get_datasets(INPUT)
    args["loader_dict"] = _create_dataloaders(config_args["dataloader"], **datasets_dict)
    args["optimizer"] = factory.import_object(config_args["optimizer"], model=args["model"])
    args["loss_function"] = factory.import_object(config_args['loss_function'])
    args["lr_scheduler"] = factory.import_object(config_args['lr_scheduler'], optimizer=args["optimizer"])
    args["observers_dict"] = _get_observers(config_args["observers"])
    args["epochs"] = config_args["epochs"]
    args["device"] = config_args["device"]
    args["amp"] = config_args["amp"]
    args["output_dir"] = OUTPUT 
    return args

if __name__ == "__main__":
    args = shared_logic.get_pipeline_stage_args(FILE_NAME)
    shared_logic.log_arguments(FILE_NAME, args)
    processed_args = prepare_pip_arguments(args)
    trainer.process(**processed_args)

