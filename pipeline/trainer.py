from pathlib import Path
from functools import partial

import torch
import ray.tune as tune

import src.utils.pipeline_repository as pipeline_repository
import src.observers.subscribers as subscribers
import src.observers.metrics as metrics
from src.hypertuner import HyperTuner
import src.observers.subscribers as subscribers 
import src.observers.metrics as metrics 

FILE_NAME = Path(__file__).stem
_MODEL_NAME = "model.pickle"
_WEIGHTS_NAME = "weights.pt"

def get_model():
    model_dict = pipeline_repository.get_objects("trainer/output")
    model, weights = model_dict["model"], model_dict["weights"]
    model.load_state_dict(weights)
    return model

def build_optimizer(model, optimizer, config):
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
    return optimizer.__class__(params)

def build_model(config: dict, model, optimizer, lr_scheduler, loss_function: torch.nn, device: str, train_loader, valid_loader):
    model = model.copy()
    cm = subscribers.Confusion_Matrix(class_num=model.num_classes, metrics=[metrics.mIoU], when="TEST")
    model.observers={"after_epoch": [], "after_step": [cm], "before_step": [], "before_epoch": []}
    model.optimizer = build_optimizer(model, optimizer, config)
    model.loss_function = loss_function
    model.scheduler = lr_scheduler.__class__(model.optimizer, T_max=lr_scheduler.T_max)
    model.device = device
    model.train_loader, model.valid_loader = train_loader, valid_loader
    return model

def optimal_model(model, optimizer, lr_scheduler, loss_function, device, hypertuner):
    model_builder = partial(build_model, 
                            model=model, 
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            loss_function=loss_function,
                            device=device, 
                            train_loader=model.train_loader, 
                            valid_loader=model.valid_loader)
    trainable = partial(_hy_trainable, iterations=hypertuner.iterations)
    trainable = tune.with_parameters(trainable, model_factory=model_builder)
    hyper_df = hypertuner.run(trainable)
    hyper_path = pipeline_repository.get_path(Path("trainer/artifacts"))
    pipeline_repository.create_dir_if_not_exist(hyper_path)
    hyper_path = hyper_path / "hyper.csv"
    hyper_df.to_csv(str(hyper_path))
    best = hypertuner.analysis.best_result["config"]
    model.optimizer = build_optimizer(model, optimizer, best)
    return model

def _hy_trainable(config, iterations, model_factory):
    tune.utils.wait_for_gpu(target_util=0)
    model = model_factory(config)
    def hook_fun(self):
        def inner():
            results = self.observer_results()
            mIoU = results["mIoU"]
            tune.report(performance=mIoU)
        return inner
    model.after_epoch_hook = hook_fun(model)
    model.fit(iterations)
    
def process(active: bool, epochs: int, amp: bool, mixup_factor: float, device: str, model, loader_dict: dict, loss_function, optimizer, lr_scheduler, observers_dict: dict, hypertuner: HyperTuner, output_dir: Path):
    model.optimizer = optimizer
    model.scheduler = lr_scheduler
    model.loss_function = loss_function
    model.train_loader = loader_dict["train"]
    if "valid" in loader_dict.keys():
        model.valid_loader = loader_dict["valid"]
    pipeline_repository.push_pickled_obj(FILE_NAME, "output", model, _MODEL_NAME)
    if hypertuner.active:
        model = optimal_model(model, optimizer, lr_scheduler, loss_function, device, hypertuner)
    model.observers = observers_dict
    model.device = device
    model.use_amp = amp
    model.mixup_factor = mixup_factor
    if active:
        model.fit(epochs=epochs)
        output_dir = pipeline_repository.create_dir_if_not_exist(output_dir)
        output_path = pipeline_repository.get_path(output_dir / _WEIGHTS_NAME)
        torch.save(model.state_dict(), str(output_path))
