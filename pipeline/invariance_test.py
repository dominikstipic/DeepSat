from pathlib import Path
import pandas as pd
from runners import invariance_test

from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository
from torch.utils.data import DataLoader

def set_attribute(dataset, attribute_transform):
    tensor_transform = dataset.transform.transforms[-1]
    compose = Compose([attribute_transform, tensor_transform])
    dataset.transform = compose

def save_analysis_to_csv(results: dict, out_dir: Path):
    identity_key, identity_perf = [(k, v) for k,v in results.items() if "Identity" in k][0]
    results.pop(identity_key)
    domain_costs = {k: (identity_perf - attribute_perf) for k, attribute_perf in results.items()}
    df = pd.DataFrame(list(domain_costs.items()), columns=["Domain Attribute", "Domain cost"])
    out_dir = pipeline_repository.create_dir_if_not_exist(out_dir)
    path = out_dir / "analysis.csv"
    df.to_csv(path)

def process(dataset, model, device: str, attributes: list, observers_dict: dict, output_dir: Path):
    model.device = device
    model.observers = observers_dict
    invariance_analysis = {}
    for attribute_name, attribute in attributes:
        attribute_name = str(attribute_name)
        set_attribute(dataset, attribute)
        dataloader = DataLoader(dataset, batch_size=1)
        model.valid_loader = dataloader
        model.evaluate()
        results = model.observer_results()
        print(f"{attribute_name}-{results}") 
        invariance_analysis[attribute_name] = results["mIoU"]
    save_analysis_to_csv(invariance_analysis, output_dir)
    

