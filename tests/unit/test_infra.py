import os
from pathlib import Path

import pytest

import src.utils.pipeline_repository as pipeline_repository

REPO = pipeline_repository.get_path("")

def run_pipline(data_path, config_path):
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
    weights1 = run_pipline(data_path, config_path)
    weights2 = run_pipline(data_path, config_path)

    for (key1, value1), (key2, value2) in zip(weights1.items(), weights2.items()):
        if key1 != key2: 
            assert False, "Keys are different"
        if (value1 != value2).any().item():
            assert False, "Values are different"
    assert True
