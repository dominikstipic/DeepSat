import pathlib
import tarfile
import io
from PIL import Image
from pathlib import Path

import torch

from src.utils.common import merge_list_2d, unpack_tar_archive_for_paths

IDENTITY = lambda x : x

class TarDataset(torch.utils.data.IterableDataset):
    mean = None
    std  = None

    def __init__(self, path: Path, transform=IDENTITY):
        super().__init__()
        path = Path(path)
        if not path.exists(): 
            raise RuntimeError("The given path doesn't exist.")
        self.path = path.resolve()
        self.tars = list(self.path.iterdir())
        self.transform = transform
        self.length = None

    def __len__(self):
        if not self.length:
            self.length = 0
            for tar in self.tars:
                with tarfile.open(str(tar), "r") as tar:
                    self.length += len(tar.getnames()) // 2
        return self.length

    def get_paths(self):
        paths_list2d = list(map(lambda shard: unpack_tar_archive_for_paths(shard), self.tars))
        paths = merge_list_2d(paths_list2d)
        paths = list(filter(lambda p: Path(p).stem.startswith("img"), paths))
        paths = sorted(paths)
        return paths

    def tar_generator(self, path: Path):
        with tarfile.open(path, "r") as tar:
            for tar_info in tar:
                file = tar.extractfile(tar_info)
                content = file.read()
                pil_image = Image.open(io.BytesIO(content))
                yield pil_image, tar_info.path
    
    def copy(self):
        path = self.path
        return TarDataset(path, self.transform)

    def get_example(self, gen):
        img, self.img_path = next(gen)
        mask, self.mask_path = next(gen)
        self.path = "-".join(Path(self.mask_path).name.split("-")[1:])
        return img, mask

    def __iter__(self):
        for tar in self.tars:
            gen = self.tar_generator(tar)
            try:
                while True:
                    img, mask = self.get_example(gen)
                    img, mask = self.transform([img, mask])
                    yield img, mask
            except StopIteration:
                pass