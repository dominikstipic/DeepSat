import os
from pathlib import Path
import pytest

import pandas as pd

import src.utils.pipeline_repository as pipeline_repository

REPO = pipeline_repository.get_path("")

def _run_pipline(data_path, config_path):
    pipeline_repository.clean()
    preprocess_path = REPO / "preprocess/output"
    pipeline_repository.create_dir_if_not_exist(preprocess_path)
    
    cp_cmd = f"cp -r {data_path + '/*'} {str(preprocess_path)}"
    run_cmd = f"python main.py --config_path={config_path} --data_path={data_path}"
    os.system(cp_cmd)
    os.system(run_cmd)
    objects = pipeline_repository.get_objects(Path("trainer/output"))
    return objects["weights"]

def test_reproducibility():
    data_path = "tests/unit/resources/reproducibility/data"
    config_path = "tests/unit/resources/reproducibility/infra.json"
    weights1 = _run_pipline(data_path, config_path)
    weights2 = _run_pipline(data_path, config_path)

    for (key1, value1), (key2, value2) in zip(weights1.items(), weights2.items()):
        if key1 != key2: 
            assert False, "Keys are different"
        if (value1 != value2).any().item():
            assert False, "Values are different"
    assert True

def _get_losses():
    path = "repository/trainer/artifacts/metrics-TRAIN.csv"
    metrics = pd.read_csv(path)
    losess = list(metrics["train"])
    return losess

def test_overfitting():
    pipeline_repository.clean()
    input_path  = "tests/unit/resources/overfit/data"
    config_path = "tests/unit/resources/overfit/infra.json" 
    dataset_for_eval = "repository/dataset_factory/output/train_db.pickle"
    dataset_factory_cmd = f"python -m runners.dataset_factory --input={input_path} --config={config_path}"
    trainer_cmd = f"python -m runners.trainer --config={config_path}"
    eval_cmd = f"python -m runners.evaluation --config={config_path} --dataset_input={dataset_for_eval}"
    os.system(dataset_factory_cmd)
    os.system(trainer_cmd)
    os.system(eval_cmd)
    losess = _get_losses()
    flag = True
    for i in range(len(losess)-1):
        descending = losess[i+1] < losess[i]
        flag = flag and descending
    assert flag, "The loss doesn't monotonically fall with epochs. This could indicate that model doesn't have enought capacity"


def test_splits():
    input = "dataset_factory/output"
    datasets_dict = pipeline_repository.get_objects(input)
    datasets = list(datasets_dict.values())
    flag = True
    for i in range(len(datasets)):
        first = datasets[0].get_paths()
        for j in range(i+1, len(datasets)):
            second = datasets[j].get_paths()
            intersection = set(first).intersection(second)
            if len(intersection) > 0:
                pytest.fail(f"{datasets[i]} and {datasets[j]} have common examples")

