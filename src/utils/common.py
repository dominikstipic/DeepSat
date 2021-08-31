from datetime import datetime
import json
import itertools
import pytz
from pathlib import Path
import tarfile
import pickle

def create_dir_if_not_exist(path: Path):
    if not path.exists():
        path.mkdir()

def merge_list_dicts(dicts: list) -> dict:
  d = {}
  for di in dicts:
    for k,v in di.items():
      d[k] = v
  return d
  
def read_json(path: Path) -> dict:
  if not path.exists:
    raise FileNotFoundError(f"Json: {path} doesn't exist")
  with open(str(path), "r") as json_file:
    json_dict = json.load(json_file)
  return json_dict

def write_json(data, path):
  with open(path, 'w') as fp:
    json.dump(data, fp)

def current_time():
  tz = pytz.timezone('CET')
  now = datetime.now(tz)
  current_time = now.strftime("%d|%m|%y, %H:%M:%S")
  return current_time

def merge_list_2d(xss: list) -> list:
  merged = list(itertools.chain(*xss))
  return merged

def unpack_tar_archive_for_paths(tar_path: Path) -> list:
  with tarfile.open(tar_path, "r:gz") as tar: 
    paths = map(lambda x : x.path, list(tar))
  return list(paths)

def save_pickle(path: Path, pickle_object: pickle):
  with open(str(path), "wb") as fp:
    pickle.dump(pickle_object, fp)

def load_pickle(path: Path) -> pickle:
  if type(path) == str: path = Path(path)
  if not path.exists():
    return []
  if not path.name.endswith(".pickle"): 
    path = path.parent / f"{path.stem}.pickle"
  with open(str(path), "rb") as fp: 
    pickle_object = pickle.load(fp)
  return pickle_object
