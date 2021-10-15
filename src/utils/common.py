from datetime import datetime
import json
import itertools
import pytz
from pathlib import Path
import tarfile
import pickle
from PIL import Image
from collections import OrderedDict


import numpy as np 
import cv2

def current_time():
  tz = pytz.timezone('CET')
  now = datetime.now(tz)
  current_time = now.strftime("%H:%M %d-%m-%y")
  return current_time

def merge_list_dicts(dicts: list) -> dict:
  d = {}
  for di in dicts:
    for k,v in di.items():
      d[k] = v
  return d

def merge_list_2d(xss: list) -> list:
  merged = list(itertools.chain(*xss))
  return merged
  
def read_json(path: Path, ordered_dict=False) -> dict:
  if not path.exists():
    return None
  with open(str(path), "r") as json_file:
    json_dict = json.load(json_file) if not ordered_dict else json.load(json_file, object_pairs_hook=OrderedDict)
  return json_dict

def write_json(data: dict, path: Path):
  with open(str(path), 'w') as fp:
    json.dump(data, fp, indent=4)

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

def h_concatenate_images(img1, img2, is_pil=True):
  if img1.size != img2.size:
    size1,_ = img1.size
    size2,_ = img2.size
    if size1 < size2: img1 = img1.resize(img2.size)
    else: img2 = img2.resize(img1.size)
  img1, img2 = np.array(img1), np.array(img2)
  if len(img1.shape) != 2 and len(img2.shape) == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
  elif len(img1.shape) == 2 and len(img2.shape) != 2: img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
  img = cv2.hconcat([img1, img2])
  img1 = cv2.copyMakeBorder(img1, 0, 0, 0, 10, cv2.BORDER_CONSTANT, None, value = 0)
  if is_pil:
    img = Image.fromarray(img )
  return img

def renorm_tensor(t, mean, std):
  X = t.clone()
  for i in range(len(t)):
    X[i] = X[i]*std[i] + mean[i]
  return X