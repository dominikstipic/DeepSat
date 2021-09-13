from pathlib import Path
import torch

import src.utils.pipeline_repository as pipeline_repository

FILE_NAME = Path(__file__).stem
_MODEL_NAME = "model.pickle"
_WEIGHTS_NAME = "weights.pt"

def get_model():
    model_dict = pipeline_repository.get_objects("trainer/output")
    model, weights = model_dict["model"], model_dict["weights"]
    model.load_state_dict(torch.load(weights))
    return model
    
def process(epochs: int, amp: bool, device: str, model, loader_dict: dict, loss_function, optimizer, lr_scheduler, observers_dict: dict, output_dir: Path):
    model.optimizer = optimizer
    model.scheduler = lr_scheduler
    model.loss_function = loss_function
    model.device = device
    model.observers = observers_dict
    pipeline_repository.push_pickled_obj(FILE_NAME, "output", model, _MODEL_NAME)

    model.train_loader = loader_dict["train_dl"]
    model.valid_loader = loader_dict["valid_dl"]

    model.fit(epochs=epochs, amp=amp)

    output_dir = pipeline_repository.create_dir_if_not_exist(output_dir)
    output_path = pipeline_repository.get_path(output_dir / _WEIGHTS_NAME)
    torch.save(model.state_dict(), str(output_path))

    

    


