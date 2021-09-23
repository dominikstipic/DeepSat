from pathlib import Path

from tqdm import tqdm

import src.utils.pipeline_repository as pipeline_repository
import src.utils.hashes as hashes

def _save_surplus_files(shard_nums: int, output_dir: Path, shard_size: int, img_names_paths: list, input_dir: Path, extension: str):
    shard_surplus_size = len(img_names_paths) % shard_size
    if shard_surplus_size != 0:
        print(f"SURPLUS SIZE = {shard_surplus_size}")
        surplus = img_names_paths[-shard_surplus_size:]
        img_names_paths = img_names_paths[:-shard_surplus_size]
        shard_path = f"{output_dir}/shard_{shard_nums}.tar"
        input_file_paths =  [input_dir / f"{name}.{extension}" for name in surplus]
        pipeline_repository.push_as_tar(input_file_paths, Path(shard_path))

def _save_other_files(shard_nums: int, output_dir: Path, shard_size: int, img_names_paths: list, input_dir: Path, extension: str):
    with tqdm(total=shard_nums) as pbar:
        for num in range(shard_nums):
            shard_path = f"{output_dir}/shard_{num}.tar"
            start, end = num*shard_size, (num+1)*shard_size
            shard_paths = img_names_paths[start:end] 
            input_file_paths =  [input_dir / f"{name}.{extension}" for name in shard_paths]
            pipeline_repository.push_as_tar(input_file_paths, Path(shard_path))
            pbar.update()

def process(shard_size: int, input_dir: Path, output_dir: Path) -> None:
    file_paths = pipeline_repository.get_obj_paths(input_dir.parent.name, input_dir.name)
    extension = str(file_paths[0]).split(".")[-1]
    get_example_name = lambda name: f"{name.split('-')[1]}-{name.split('-')[2]}"
    img_names = [get_example_name(name.stem) for name in file_paths]
    img_names = list(filter(lambda x : not x.endswith(f".{extension}"), img_names))
    img_names = hashes.random_shuffle(img_names)
    img_names_paths = sorted(list(set([str(name) for name in img_names])))
    shard_nums = len(img_names_paths) // shard_size
    _save_surplus_files(shard_nums, output_dir, shard_size, img_names_paths, input_dir, extension)
    _save_other_files(shard_nums, output_dir, shard_size, img_names_paths, input_dir, extension)


    
    

