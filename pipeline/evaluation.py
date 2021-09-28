from pathlib import Path
from torch.utils.data import DataLoader

import src.utils.pipeline_repository as pipeline_repository

FILE_NAME = Path(__file__).stem
_OUT_NAME = "metrics.json"

def process(model, device: str, test_ld: DataLoader, observers_dict: dict, output_dir: Path):
    model.device = device
    model.observers = observers_dict
    model.valid_loader = test_ld
    model.evaluate()
    results = model.observer_results()
    print(results)

    output_dir = pipeline_repository.create_dir_if_not_exist(output_dir)
    pipeline_repository.push_json(output_dir, _OUT_NAME, results)




