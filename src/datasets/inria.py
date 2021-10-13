from PIL import Image
import pathlib 

from src.sat_dataset import Sat_Dataset

class Inria(Sat_Dataset):
    mean = [103.2342, 108.9520, 100.1419]
    std  = [48.6281, 44.4967, 41.9143]

    def __init__(self, root, transforms=None):
        super().__init__(root=root, transforms=transforms, split=None)
        self.root = pathlib.Path(root)
        index = [x for x in list(self.root.iterdir())]
        self.data = [name for name in index if not str(name).endswith("-mask.tif")]
        self.labels = [path.parent / f"{path.stem}-mask.tif" for path in self.data]

    def get_paths(self):
        paths = [str(p) for p in self.data]
        paths = sorted(paths)
        return paths

    def get_examples(self):
        return self.data

    def get(self, idx):
        img_name, label_name = self.data[idx], self.labels[idx]
        img, mask = Image.open(img_name), Image.open(label_name)
        return img, mask 

    def copy(self):
        return Inria(self.root, self.transforms)