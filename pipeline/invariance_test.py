from pathlib import Path
import pandas as pd

from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository

def set_attribute(dataset, attribute_transform):
    tensor_transform = dataset.transform.transforms[-1]
    compose = Compose([attribute_transform, tensor_transform])
    dataset.transform = compose


def process(dataset, model, attributes: list, output_dir: Path):
    results = {}
    for attribute in attributes:
        attribute_name = type(attribute).__name__
        set_attribute(dataset, attribute)
        model.evaluate()
        results = model.observer_results()
        print(f"{attribute_name}-{results}") 
        results[attribute_name] = results