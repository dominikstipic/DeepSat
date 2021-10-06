from pathlib import Path
import itertools
import pandas as pd

import torch

import src.utils.pipeline_repository as pipeline_repository

FILE_NAME = Path(__file__).stem
_MODEL_NAME = "model.pickle"
_WEIGHTS_NAME = "weights.pt"

def get_model():
    model_dict = pipeline_repository.get_objects("trainer/output")
    model, weights = model_dict["model"], model_dict["weights"]
    model.load_state_dict(weights)
    return model

############################
def get_params(model, lr1, lr2, wd1, wd2):
    return [
                {
                    "params": model.random_init_params(), 
                    "lr": lr1, 
                    "weight_decay": wd1
                }, 
                {
                    "params": model.fine_tune_params(), 
                    "lr": lr2, 
                    "weight_decay": wd2
                }
            ]

def hiperparameter_optimization(model, hiper_optim, amp):
    builder = torch.optim.Adam
    lr_1, lr_2 = hiper_optim["lr_1"], hiper_optim["lr_2"]
    weight_decay_1, weight_decay_2 = hiper_optim["weight_decay_1"], hiper_optim["weight_decay_2"]
    xs = [lr_1, lr_2, weight_decay_1, weight_decay_2]
    results = []
    for lr1, lr2, wd1, wd2 in itertools.product(*xs):
        params = get_params(model, lr1, lr2, wd1, wd2)
        model.optimizer = builder(params = params)
        
        model.use_amp = amp
        model.train_state()
        for epoch in range(hiper_optim["epochs"]): 
            model.outer_state.epoch = epoch
            model.one_epoch()
        model.evaluate()
        loss = model.observer_results()["loss"]
        item = [(lr1, lr2, wd1, wd2), loss]
        results.append(item)
    best_param = get_params(model, *min(results, key=lambda xs : xs[1])[0])
    best_optimizer = builder(params = best_param)
    model.optimizer = best_optimizer
    return model, results

def save_hiper_results(results: list, output: Path): 
    output = pipeline_repository.create_dir_if_not_exist(output) / "hiper_search.csv"
    results = [[*xs[0], xs[1]] for xs in results]
    df = pd.DataFrame(results)
    df.to_csv(str(output), index=False, header=False)
############################

def process(epochs: int, amp: bool, hiper_optim: dict, device: str, model, loader_dict: dict, loss_function, optimizer, lr_scheduler, observers_dict: dict, output_dir: Path):
    model.optimizer = optimizer
    model.scheduler = lr_scheduler
    model.loss_function = loss_function
    model.device = device
    model.observers = observers_dict
    pipeline_repository.push_pickled_obj(FILE_NAME, "output", model, _MODEL_NAME)

    model.train_loader = loader_dict["train"]
    if "valid" in loader_dict.keys():
        model.valid_loader = loader_dict["valid"]

    if hiper_optim["activate"]:
        model, results = hiperparameter_optimization(model, hiper_optim, amp)
        results_dir = pipeline_repository.get_path("trainer/artifacts")
        save_hiper_results(results, results_dir)

    model.fit(epochs=epochs, amp=amp)
    output_dir = pipeline_repository.create_dir_if_not_exist(output_dir)
    output_path = pipeline_repository.get_path(output_dir / _WEIGHTS_NAME)
    torch.save(model.state_dict(), str(output_path))

    

    


