import pathlib
import tarfile
import io
from PIL import Image
from pathlib import Path

import torch

from src.utils.common import merge_list_2d, unpack_tar_archive_for_paths

IDENTITY = lambda x : x

class TarDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: Path, transform=IDENTITY):
        super().__init__()
        self.tars = list(pathlib.Path(path).iterdir())
        self.transform = transform

    def __len__(self):
        with tarfile.open(str(self.tars[0]), "r") as tar:
            tar_size = len(tar.getnames()) // 2
        return len(self.tars) * tar_size

    def get_paths(self):
        paths_list2d = list(map(lambda shard: unpack_tar_archive_for_paths(shard), self.tars))
        paths = merge_list_2d(paths_list2d)
        return paths

    def tar_generator(self, path):
        with tarfile.open(path, "r") as tar:
            for tar_info in tar:
                file = tar.extractfile(tar_info)
                content = file.read()
                pil_image = Image.open(io.BytesIO(content))
                yield pil_image
    
    def copy(self):
        path = self.tars[0].parent
        return TarDataset(path, self.transform)

    def __iter__(self):
        for tar in self.tars:
            gen = self.tar_generator(str(tar))
            try:
                while True:
                    img = next(gen)
                    mask = next(gen)
                    img, mask = self.transform([img, mask])
                    yield img, mask
            except StopIteration:
                pass