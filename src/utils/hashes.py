import hashlib
import random

from pathlib import Path

def current_data_hash():
    paths = [str(p) for p in Path("data").iterdir()]
    path_strs = " ".join(paths)
    hex = hashlib.md5(path_strs.encode()).hexdigest()
    return hex

def random_shuffle(str_list: list):
    data  = " ".join(str_list)
    digit = from_string(data, len(data))
    random.seed(digit)
    random.shuffle(str_list) 
    return str_list

def get_digest(data: str):
    hash_object = hashlib.md5(data.encode()).hexdigest()
    return hash_object

def from_string(data: str, scale: int) -> int:
    hash_object = hashlib.md5(data.encode()).hexdigest()
    digit = int(hash_object, 16) % scale
    return digit
    
class HashGenerator:
    def __init__(self, data: str, scale: int):
        self.num = 0
        self.scale = scale
        self.data = data

    def __iter__(self):
        return self

    def __next__(self) -> int:
        data = self.data + str(self.num)
        result = from_string(data, self.scale)
        self.num += 1
        return result
    
    def sample(self, size: int, unique=True) -> list:
        if unique: assert size <= self.scale, "Cannot generate unique sample with given sample size"
        arr = []
        while len(arr) < size:
            sample = next(self)
            if unique:
                if sample not in arr: 
                    arr.append(sample)
            else:
                arr.append(sample)
        return arr

