from pathlib import Path
import torch

import src.utils.pipeline_repository as pipeline_repository

FILE_NAME = Path(__file__).stem
_MODEL_NAME = "model.pt"

def process(epochs: int, amp: bool, device: str, model, loader_dict: dict, loss_function, optimizer, lr_scheduler, observers_dict: dict, output_dir: Path):
    model.train_loader = loader_dict["train_dl"]
    model.valid_loader = loader_dict["valid_dl"]
    model.optimizer = optimizer
    model.scheduler = lr_scheduler
    model.loss_function = loss_function
    model.device = device
    model.observers = observers_dict

    model.fit(epochs=epochs, amp=amp)

    pipeline_repository.create_dir(output_dir)
    output_path = pipeline_repository.get_path(output_dir / _MODEL_NAME)
    torch.save(model, str(output_path))

    pipeline_repository.push_pickled_obj(FILE_NAME, "output", model, "model")



