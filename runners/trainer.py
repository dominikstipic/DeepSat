from pathlib import Path
import copy

from . import shared_logic as shared_logic
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import src.observers.metrics as metrics
import pipeline.trainer as trainer

FILE_NAME = Path(__file__).stem
INPUT  = Path("dataset_factory/output")
OUTPUT = Path(f"{FILE_NAME}/output") 

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

def prepare_pip_arguments(config_args: dict):
    global INPUT, OUTPUT
    args = {} 
    args["model"] = factory.import_object(config_args['model'])
    datasets_dict = _get_datasets(INPUT)
    args["loader_dict"] = _create_dataloaders(config_args["dataloader"], **datasets_dict)
    args["optimizer"] = factory.import_object(config_args["optimizer"], model=args["model"])
    args["loss_function"] = factory.import_object(config_args['loss_function'])
    args["lr_scheduler"] = factory.import_object(config_args['lr_scheduler'], optimizer=args["optimizer"])
    args["observers_dict"] = get_observers(config_args["observers"])
    args["epochs"] = config_args["epochs"]
    args["device"] = config_args["device"]
    args["amp"] = config_args["amp"]
    args["output_dir"] = OUTPUT 
    return args

if __name__ == "__main__":
    args = shared_logic.prerun_routine(FILE_NAME)
    processed_args = prepare_pip_arguments(args)
    trainer.process(**processed_args)

