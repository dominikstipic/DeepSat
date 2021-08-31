"""
    Cleans the disk resources
"""

import json
import shutil
from pathlib import Path

from src.utils.storage import Storage
from src.init_script import config

cache_path = Path(config["CACHE_DIR"])
image_path = Path(config["IMAGE_DIR"])
print(cache_path, image_path)

Storage(cache_path)
Storage.get().clear()

shutil.rmtree(image_path)
image_path.mkdir()

