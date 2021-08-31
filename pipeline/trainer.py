import os
import argparse
from pathlib import Path

from src.utils.common import load_pickle

def cmi_parse() -> tuple:
    parser = argparse.ArgumentParser(description="The model training stage of the pipeline")
    parser.add_argument("epochs", type=int, help="number of epochs to train")
    parser.add_argument("device", type=str, help="device where the model will be trained")
    parser.add_argument("model", type=str, help="path to model pickle object")
    parser.add_argument("loss_function", type=str, help="path to loss function pickle object")
    parser.add_argument("dataloader", type=str, help="path to dataloader dir")
    parser.add_argument("optimizer", type=str, help="path to optimizer pickle object")
    parser.add_argument("lr_scheduler", type=str, help="path to scheduler pickle object")
    parser.add_argument("observers", type=str, help="path to observers dir")
    args = parser.parse_args()
    args.model = load_pickle(args.model)

    train_dl = load_pickle(args.dataloader + "/train_dl")
    valid_dl = load_pickle(args.dataloader + "/valid_dl")
    test_dl = load_pickle(args.dataloader + "/test_dl")
    args.loader_dict = dict(train_dl=train_dl, valid_dl=valid_dl, test_dl=test_dl)

    args.loss_function = load_pickle(args.loss_function)
    args.optimizer = load_pickle(args.optimizer)
    args.lr_scheduler = load_pickle(args.lr_scheduler)

    obs_dir = Path(args.observers)
    get_paths  = lambda event_name: sorted([str(p) for p in list((obs_dir / event_name).iterdir())], key=lambda s: int(Path(s).stem))
    import_obs = lambda obs_path_list: list(map(lambda str_path: load_pickle(str_path), obs_path_list))
    after_epoch_obs = import_obs(get_paths("after_epoch"))
    before_epoch_obs = import_obs(get_paths("before_epoch"))
    after_step_obs = import_obs(get_paths("after_step"))
    before_step_obs = import_obs(get_paths("before_step"))
    args.observers_dict = {"after_epoch": after_epoch_obs, 
                           "before_epoch": before_epoch_obs,
                           "after_step": after_step_obs, 
                           "before_step": before_step_obs}
    return vars(args)

def process(epochs: int, amp: bool, device: str, model, loader_dict: dict, loss_function, optimizer, lr_scheduler, observers_dict: dict, **kwargs):
    model.train_loader = loader_dict["train_dl"]
    model.valid_loader = loader_dict["valid_dl"]

    model.optimizer = optimizer
    model.scheduler = lr_scheduler
    model.loss_function = loss_function
    model.device = device

    model.observers = observers_dict
    model.fit(epochs=epochs, amp=amp)


if __name__ == "__main__":
    args = cmi_parse()
    process(**args)