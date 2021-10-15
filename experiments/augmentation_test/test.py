from pathlib import Path
from itertools import chain, combinations
import os

import src.utils.common as common
import src.utils.pipeline_repository as pipeline_repository

def create_dir_if_not_exist(root_dir: Path) -> Path:
    if not root_dir.exists(): 
        os.makedirs(str(root_dir), exist_ok=True)
    return root_dir

def get_augmentations(config: dict):
    return config["dataset_factory"]["augmentations"]["train"].copy()

def get_config():
    CONFIG_JSON = Path("experiments/augmentation_test/exp.json")
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

def process(config_dir: Path, out_dir: Path):
    config = get_config()
    augmentation_list = get_augmentations(config)
    create_configs(augmentation_list, config, config_dir)
    pipeline_repository.clean()
    for config_path in config_dir.iterdir():
        print(out_dir)
        run_cmd = f"python main.py --config_path={str(config_path)}"
        os.system(run_cmd)
        out_path = str(out_dir / config_path.stem)
        eval_cp_cmd  = f"cp -r repository/evaluation {out_path}"
        train_cp_cmd = f"cp -r repository/trainer {out_path}"
        os.system(eval_cp_cmd), os.system(train_cp_cmd)


if __name__ == "__main__":
    CONFIG_DIR = Path("experiments/augmentation_test/configs")
    CONFIG_DIR = create_dir_if_not_exist(CONFIG_DIR)
    OUT_DIR = Path("experiments/augmentation_test/out")
    OUT_DIR = create_dir_if_not_exist(OUT_DIR)
    process(CONFIG_DIR, OUT_DIR)
