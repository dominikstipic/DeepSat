import os
import src.utils.pipeline_repository as pipeline_repository

REPO = pipeline_repository.get_path("")

def test_reproducibility():
    pipeline_repository.clean()
    preprocess_path = REPO / "preprocess/output"
    pipeline_repository.create_dir_if_not_exist(preprocess_path)
    data_path = "tests/unit/resources/reproducibility/data"
    config_path = "tests/unit/resources/reproducibility/infra.json"

    cp_cmd = f"cp -r {data_path + '/*'} {str(preprocess_path)}"
    run_cmd = f"python main.py --config_path={config_path} --data_path={data_path}"
    os.system(cp_cmd)
    os.system(run_cmd)


test_reproducibility()