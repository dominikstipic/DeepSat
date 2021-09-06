from pathlib import Path
import csv
import os
import tarfile
import pickle

from src.datasets.tar_dataset import IDENTITY
from src.utils import common
from src.utils.common import load_pickle, save_pickle

PIPELINE_REPO = Path("repository")
IDENTITY = lambda x : x

def get_path(str_path: str) -> Path:
    str_path = str(str_path)
    if not str_path.startswith(str(PIPELINE_REPO)):
        return PIPELINE_REPO / str_path
    return Path(str_path)

def create_dir_if_not_exist(root_dir: Path) -> Path:
    root_dir = get_path(str(root_dir))
    if not root_dir.exists(): 
        os.makedirs(str(root_dir), exist_ok=True)
    return root_dir

##########################3

def push_csv(dir_path: Path, name: str, csv_header: list, data, append=False, write_function=IDENTITY) -> None:
    dir_path = get_path(dir_path)
    create_dir_if_not_exist(dir_path)
    name = name + ".csv" if not name.endswith(".csv") else name
    out_path = dir_path / name 
    flag = "w" if not append else "a" 
    with open(out_path, flag) as csvfile: 
        csvwriter = csv.writer(csvfile) 
        if not append: 
            csvwriter.writerow(csv_header) 
        for example in data:
            to_write = write_function(example)
            csvwriter.writerow(to_write)

def push_images(artifact_home_dir: Path, images: list, names: list):
    if not artifact_home_dir.exists(): 
        artifact_home_dir=create_dir_if_not_exist(artifact_home_dir)
    for img_name, img in zip(names, images):
        img_path = artifact_home_dir / img_name 
        img.save(str(img_path))

def push_as_tar(input_file_paths: list, tar_output_path: Path) -> None:
    global PIPELINE_REPO
    input_file_paths = [PIPELINE_REPO / p for p in input_file_paths]

    tar_name = tar_output_path.name
    tar_output_path = create_dir_if_not_exist(tar_output_path.parent)
    tar_output_path = tar_output_path / tar_name

    with tarfile.open(str(tar_output_path), "w:gz") as tar: 
        for f in input_file_paths: 
            img_path  = f.parent / f"img-{f.name}"
            mask_path = f.parent / f"mask-{f.name}"
            tar.add(str(img_path))
            tar.add(str(mask_path))

def push_pickled_obj(pipeline_stage_name: str, pickle_dir: Path, pickle_object, pickle_name) -> Path:
    out_path = create_dir_if_not_exist(Path(pipeline_stage_name) / pickle_dir)
    out_path = out_path / f"{pickle_name}.pickle"
    save_pickle(out_path, pickle_object)
    return out_path

def push_json(path_dir: Path, name: str, dictionary: dict):
    path_dir = get_path(path_dir)

    name = f"{name}.json" if not name.endswith(".json") else name
    path = create_dir_if_not_exist(path_dir) / name
    common.write_json(dictionary, path)

##################################

def get_obj_paths(pipeline_stage_name: str, root_dir: Path):
    path = get_path(Path(pipeline_stage_name) / root_dir)
    return list(path.iterdir())


############################################

def get_object(repo_path: Path) -> pickle:
    path = get_path(Path(repo_path))
    pickle_obj = load_pickle(path)
    return pickle_obj

def get_objects_from_repo(repo_dir: Path) -> dict:
    path = get_path(Path(repo_dir))
    path = create_dir_if_not_exist(path)
    result_dict = {}
    for obj_path in path.iterdir():
        pickle_obj = get_object(obj_path)
        result_dict[obj_path.stem] = pickle_obj
    return result_dict

