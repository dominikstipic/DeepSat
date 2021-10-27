from pathlib import Path
import pandas as pd

from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository
from torch.utils.data import DataLoader

def set_attribute(dataset, attribute_transform):
    tensor_transform = dataset.transform.transforms[-1]
    compose = Compose([attribute_transform, tensor_transform])
    dataset.transform = compose


def process(dataset, model, device: str, attributes: list, observers_dict: dict, output_dir: Path):
    model.device = device
    model.observers = observers_dict
    results = {}
    for attribute in attributes:
        set_attribute(dataset, attribute)
        dataloader = DataLoader(dataset, batch_size=1)
        model.valid_loader = dataloader
        model.evaluate()

        results = model.observer_results()
        attribute_name = type(attribute).__name__
        print(f"{attribute_name}-{results}") 
        results[attribute_name] = results