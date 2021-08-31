"""
    Relabels the pixels into given labels according to class_to_pxds dictionary
"""

#!/usr/bin/env python3
from pathlib import Path
from PIL import Image as pimg
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class_to_pxds = {
    'Buliding': {'Building': (128, 0, 0)},
    'Tree': {'Tree': (128, 128, 0)},
    'Sky': {'Sky': (128, 128, 128)},
    'Vehicle': {'Car': (64, 0, 128), 'Truck_Bus': (192, 128, 192), 
                'Train': (192, 64, 128), 'SUVPickupTruck': (64, 128, 192)},
    'SignSymbol': {'SignSymbol': (192, 128, 128)},
    'Road': {'Road': (128, 64, 128), 'LaneMkgsDriv': (128, 0, 192),
             'LaneMkgsNonDriv': (192, 0, 64), 'RoadShoulder': (128, 128, 192)},
    'Pedestrian': {'Pedestrian': (64, 64, 0), 'Child': (192, 128, 64)},
    'Fence': {'Fence': (64, 64, 128)},
    'Pole': {'Column_Pole': (192, 192, 128)},
    'Sidewalk': {'Sidewalk': (0, 0, 192)},
    'Bicyclist': {'Bicyclist': (0, 128, 192)},
    'Void': {'Void': (0, 0, 0)}
    }

pxd_to_label = {
    (128, 0, 0) : 0,
    (128, 128, 0) : 1,
    (128, 128, 128) : 2,
    (64, 0, 128) : 3, 
    (192, 128, 192) : 3, 
    (192, 64, 128) : 3,
    (64, 128, 192) : 3,
    (192, 128, 128) : 4,
    (128, 64, 128) : 5, 
    (128, 0, 192) : 5,
    (192, 0, 64) : 5, 
    (128, 128, 192) : 5,
    (64, 64, 0) : 6,
    (192, 128, 64) : 6,
    (64, 64, 128) : 7,
    (192, 192, 128) : 8,
    (0, 0, 192) : 9,
    (0, 128, 192) : 10
    }

def pxds_to_label(image, pxd_to_label, void_label=-1):
    label = np.array(image, dtype=np.int32)
    h,w,c = label.shape
    base_nums = 256 ** np.arange(c)
    base_nums = base_nums[::-1]
    coded_dict = {}
    for k,v in pxd_to_label.items():
        pxd = np.array(k, dtype=np.int32)
        code = pxd.dot(base_nums).item()
        coded_dict[code] = v
    
    label = label.reshape(-1,3)
    coded_label = label.dot(base_nums)
    coded_label = [coded_dict.get(code, void_label) for code in coded_label]
    coded_label = np.array(coded_label, dtype=np.int32)
    coded_label = coded_label.reshape(h,w)
    return coded_label
    
def labeling_procedure(paths, pxd_to_label):
    def create_new_dir(path):
        parent_dir = path.parent
        name = path.stem + "_new"
        new_dir = parent_dir / name
        if new_dir.exists():
            print(f"Deleting path tree {new_dir}")
            shutil.rmtree(new_dir)
        print(f"Creating new dir: {new_dir}")
        new_dir.mkdir()
        return new_dir
    
    for path in paths:
        path = Path(path)
        new_dir = create_new_dir(path) 
        files = list(path.iterdir())
        for file in tqdm(files, total=len(files), leave=False, position=0):
            gt = pimg.open(file)
            labeled_gt = pxds_to_label(gt, pxd_to_label)
            #labeled_gt = pimg.fromarray(labeled_gt)
            file_name = file.stem
            file_new_path = new_dir / file_name
            np.save(file_new_path, labeled_gt)
            #labeled_gt.save(file_new_path)
    

dir_paths = ["datasets/CamVid-master/gt"]
labeling_procedure(dir_paths, pxd_to_label)

#####################

# path = "/home/doms/git/DL_project/datasets/CamVid-master/gt/0001TP_006720_L.png"
# img = pimg.open(path)
# Y = pxds_to_label(img, pxd_to_label)
# print(np.unique(Y))

