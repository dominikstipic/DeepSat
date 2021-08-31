import tarfile
import argparse
import random
import itertools
from pathlib import Path

from tqdm import tqdm

import src.utils.pipeline_repository as pipeline_repository
import src.utils.hashes as hashes

FILE_NAME = Path(__file__).stem


def cmi_parse() -> tuple:
    parser = argparse.ArgumentParser(description= "The sharding stage of the pipeline. \
                                               This stage creates shards with specified number of examples in specified directory.")
    parser.add_argument("shard_size", type=int, help="number of examples in each shard")
    parser.add_argument("--input", default=f"preprocess/output", help="input data directory")
    parser.add_argument("--output", default=f"{FILE_NAME}/output", help="output data directory")
    args = parser.parse_args()

    input_dir   = args.input
    output_dir  = args.output
    shard_size = args.shard_size
    return shard_size, Path(input_dir), Path(output_dir)

def save_shard(input_dir: Path, file_names: list, shard_path: list, extension: str) -> None:
    file_names = [[str(input_dir / f"img-{f}.{extension}"), str(input_dir / f"mask-{f}.{extension}")] for f in file_names]
    file_names = list(itertools.chain(*file_names))
    with tarfile.open(shard_path, "w:gz") as tar: 
        for f in file_names: 
            tar.add(f)

def process(shard_size: int, input_dir: Path, output_dir: Path) -> None:
    file_paths = pipeline_repository.get_obj_paths(input_dir.parent.name, input_dir.name)
    extension = str(file_paths[0]).split(".")[-1]
    get_example_name = lambda name: f"{name.split('-')[1]}-{name.split('-')[2]}"
    img_names = [get_example_name(name.stem) for name in file_paths]
    img_names = list(filter(lambda x : not x.endswith(f".{extension}"), img_names))
    img_names = hashes.random_shuffle(img_names)
    img_names_paths = list(set([str(name) for name in img_names]))

    # Save surplus files
    shard_nums = len(img_names_paths) // shard_size
    shard_surplus_size = len(img_names_paths) % shard_size
    if shard_surplus_size != 0:
        print(f"SURPLUS SIZE = {shard_surplus_size}")
        surplus = img_names_paths[-shard_surplus_size:]
        img_names_paths = img_names_paths[:-shard_surplus_size]
        shard_path = f"{output_dir}/shard_{shard_nums}.tar"
        input_file_paths =  [input_dir / f"{name}.{extension}" for name in surplus]
        pipeline_repository.push_as_tar(input_file_paths, Path(shard_path))

    # Group the rest of the files
    with tqdm(total=shard_nums) as pbar:
        for num in range(shard_nums):
            shard_path = f"{output_dir}/shard_{num}.tar"
            start, end = num*shard_size, (num+1)*shard_size
            shard_paths = img_names_paths[start:end] 
            input_file_paths =  [input_dir / f"{name}.{extension}" for name in shard_paths]
            pipeline_repository.push_as_tar(input_file_paths, Path(shard_path))
            pbar.update()

if __name__ == "__main__":
    shard_size, input_dir, output_dir = cmi_parse()
    process(shard_size, Path(input_dir), Path(output_dir))
    
    

