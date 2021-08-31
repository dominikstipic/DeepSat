"""
    Vizualizes predicitons in resources/images/predicton folder
"""
import re
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

def group_dir(xs, groups):
    groups = list(groups)
    grouped = dict()
    for i, group in enumerate(groups):
        grouped[i] = []
        for x in xs:
            if group in x:
                grouped[i].append(x)
    return grouped

def resources_predictions():
    """
        Groups the prediciton images from resources/images/predictions
    """
    path = Path("/content/drive/MyDrive/DSLearn/resources/images/predictions")
    # if not path.exists():
    #     path.mkdir()
    # else:
    #     shutil.rmtree(path)
    target = os.listdir(str(path))
    #import pdb
    #pdb.set_trace()
    target = [str(path / t) for t in target]
    names = [[xs for xs in t.split(",")] for t in target]
    names = [[x.strip(), y.strip()] for x,y in names]

    epochs = [n[0] for n in names]
    iters = [n[1] for n in names]

    grouped = group_dir(target, set(iters))
    return grouped

def resources_detections():
    """
        Groups the prediciton images from resources/images/detections
    """
    target = "/content/drive/MyDrive/DSLearn/resources/images/detections"
    path = Path(target)
    target = list(path.iterdir())
    target = [str(t) for t in target]
    return target

#####  VIZUALIZATION
video_name = 'video1.avi'
#images = resources_predictions()
#images = images[1]
images = resources_detections()

frame = cv2.imread(images[0])
height, width, layers = frame.shape
print(height, width, layers)

writer = cv2.VideoWriter(video_name, 0, 1, (width,height))
for image in images:
    writer.write(cv2.imread(image))

cv2.destroyAllWindows()
writer.release()

