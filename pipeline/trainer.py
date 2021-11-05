from pathlib import Path

import torch
import ray.tune as tune

import src.utils.pipeline_repository as pipeline_repository
import src.observers.subscribers as subscribers
import src.observers.metrics as metrics
from src.hypertuner import HyperTuner


FILE_NAME = Path(__file__).stem
_MODEL_NAME = "model.pickle"
_WEIGHTS_NAME = "weights.pt"

def get_model():
    model_dict = pipeline_repository.get_objects("trainer/output")
    model, weights = model_dict["model"], model_dict["weights"]
    model.load_state_dict(weights)
    return model

def build_model(model, optimizer, config: dict):
    cm = subscribers.Confusion_Matrix(class_num=2, metrics=[metrics.mIoU])
    model.observers={"after_epoch": [], "after_step": [cm], "before_step": [], "before_epoch": []}
    params = [
              {"params": model.random_init_params(), 
               "lr": config["lr1"],
               "weight_decay": config["wd1"],
               "betas": (config["beta11"], config["beta12"])},
              {"params": model.fine_tune_params(), 
               "lr": config["lr2"],
               "weight_decay": config["wd2"],
               "betas": (config["beta21"], config["beta22"])}
             ]
    new_optimizer = optimizer.__class__(params)
    model.optimizer = new_optimizer

def process(epochs: int, amp: bool, mixup_factor: float, device: str, model, loader_dict: dict, loss_function, optimizer, lr_scheduler, observers_dict: dict, hypertuner: HyperTuner, output_dir: Path):
    def _hy_trainable(config, checkpoint_dir=None):
        build_model(model, optimizer, config)
        model.train_loader, model.valid_loader = loader_dict["train"], loader_dict["valid"]
        model.train_state()
        for _ in range(1):
            model.one_epoch()
            model.evaluate()
            results = model.observer_results()
            mIoU = results["mIoU"]
            tune.report(performance=mIoU)
    model.optimizer = optimizer
    model.scheduler = lr_scheduler
    model.loss_function = loss_function
    model.train_loader = loader_dict["train"]
    if "valid" in loader_dict.keys():
        model.valid_loader = loader_dict["valid"]
    model.device = "cpu"
    pipeline_repository.push_pickled_obj(FILE_NAME, "output", model, _MODEL_NAME)
    model.device = device
    model.use_amp = amp
    model.mixup_factor = mixup_factor
    if hypertuner.active:
        hyper_df = hypertuner.run(_hy_trainable)
        hyper_path = pipeline_repository.get_path("trainer/artifacts") / "hyper.csv"
        hyper_df.to_csv(str(hyper_path))
        best = hypertuner.analysis.best_result["config"]
        model = build_model(model, optimizer, best)
    model.observers = observers_dict
    model.fit(epochs=epochs)
    output_dir = pipeline_repository.create_dir_if_not_exist(output_dir)
    output_path = pipeline_repository.get_path(output_dir / _WEIGHTS_NAME)
    torch.save(model.state_dict(), str(output_path))
