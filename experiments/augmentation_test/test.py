from pathlib import Path
from itertools import chain, combinations
import os

import src.utils.common as common
import src.utils.pipeline_repository as pipeline_repository

def create_dir_if_not_exist(root_dir: Path) -> Path:
    if not root_dir.exists(): 
        os.makedirs(str(root_dir), exist_ok=True)
    return root_dir

def get_augmentations():
    AUG_JSON = Path("experiments/augmentation_test/augs.json")
    augs = common.read_json(AUG_JSON)
    augmentation_list = augs["augs"]
    return augmentation_list

def get_config():
    CONFIG_JSON = Path("config.json")
    return common.read_json(CONFIG_JSON)

def all_subsets(ss):
    return list(chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1))))

def create_configs(augmentation_list: list, config: dict, out_dir: Path):
    indices = list(range(len(augmentation_list)))
    combinations = all_subsets(indices)
    for idx, comb in enumerate(combinations):
        aug_subset = [augmentation_list[idx] for idx in comb]
        out_json = config.copy()
        out_json["dataset_factory"]["augmentations"]["train"] = aug_subset
        out_path = out_dir / f"{idx}.json"
        common.write_json(out_json, out_path)

def process(out_dir: Path):
    augmentation_list = get_augmentations()
    config = get_config()
    create_configs(augmentation_list, config, out_dir)
    pipeline_repository.clean()
    for out_path in out_dir.iterdir():
        cmd = f"python main.py --config_path={str(out_path)}"
        os.system(cmd)

if __name__ == "__main__":
    OUT = Path("experiments/augmentation_test/configs")
    OUT = create_dir_if_not_exist(OUT)
    process(OUT)
