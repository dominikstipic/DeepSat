from pathlib import Path
import csv
import os
import tarfile
import pickle
import torch
import cv2
import shutil

from src.datasets.tar_dataset import IDENTITY
from src.utils import common
from src.utils.common import load_pickle, save_pickle

PIPELINE_REPO = Path("repository")
IDENTITY = lambda x : x

CSV_EXT = "csv"
JSON_EXT = "json"
PICKLE_EXT = "pickle"
PNG_EXT = "png"
PT_EXT = "pt"

def clean():
    shutil.rmtree(PIPELINE_REPO)
    PIPELINE_REPO.mkdir()

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

def _append_extension(name, ext):
    name = str(name)
    return f"{name}.{ext}" if not name.endswith(ext) else name

##########################3

def push_csv(dir_path: Path, name: str, csv_header: list, data, append=False, write_function=IDENTITY) -> None:
    dir_path = get_path(dir_path)
    create_dir_if_not_exist(dir_path)
    name = _append_extension(name, CSV_EXT)
    out_path = dir_path / name 
    flag = "w" if not append else "a" 
    with open(out_path, flag) as csvfile: 
        csvwriter = csv.writer(csvfile) 
        if not append: 
            csvwriter.writerow(csv_header) 
        for example in data:
            to_write = write_function(example)
            csvwriter.writerow(to_write)


def push_images(artifact_home_dir: Path, images: list, names=None):
    if not artifact_home_dir.exists(): 
        artifact_home_dir=create_dir_if_not_exist(artifact_home_dir)
    iter_list = enumerate(images) if not names else zip(names, images)
    for img_name, img in iter_list:
        img_name = _append_extension(img_name, PNG_EXT)
        img_path = artifact_home_dir / img_name
        img.save(str(img_path))


def push_as_tar(input_file_paths: list, tar_output_path: Path) -> None:
    input_file_paths = [get_path(p) for p in input_file_paths]

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
    pickle_name = _append_extension(pickle_name, PICKLE_EXT)
    out_path = out_path / pickle_name
    save_pickle(out_path, pickle_object)
    return out_path

def push_json(path_dir: Path, name: str, dictionary: dict):
    path_dir = get_path(path_dir)
    name = _append_extension(name, JSON_EXT)
    path = create_dir_if_not_exist(path_dir) / name
    common.write_json(dictionary, path)

##################################

def get_obj_paths(pipeline_stage_name: str, root_dir: Path):
    path = get_path(Path(pipeline_stage_name) / root_dir)
    return list(path.iterdir())

def get_pickle(repo_path: Path) -> pickle:
    path = get_path(Path(repo_path))
    pickle_obj = load_pickle(path)
    return pickle_obj

############################################

def _read_file(path: Path):
    path = str(path)
    endswith = lambda ext: path.endswith(ext)
    if endswith(PICKLE_EXT):
        obj = get_pickle(path)
    elif endswith(JSON_EXT):
        obj = common.read_json(Path(path))
    elif endswith(CSV_EXT):
        pass
    elif endswith(PT_EXT):
        try: obj = torch.load(path)
        except RuntimeError:
            mess = """
                      Couldn't load weight because this device doesn't support cuda. 
                      Transfering weights to cpu. 
                   """
            print(mess)
            obj = torch.load(path, map_location=torch.device("cpu"))
    elif endswith(PNG_EXT):
        obj = cv2.imread(path)
    else:
        raise RuntimeError("unknown file extension")
    return obj

def get_objects(repo_dir: Path) -> dict:
    path = get_path(Path(repo_dir))
    path = create_dir_if_not_exist(path)
    result_dict = {}
    for obj_path in path.iterdir():
        if obj_path.is_dir():
            obj = [_read_file(p) for p in obj_path.iterdir()]
        else: 
            obj = _read_file(obj_path)
        result_dict[obj_path.stem] = obj
    return result_dict

