from tqdm import tqdm
import argparse
from pathlib import Path

from torch.utils.data import Dataset

from src.transforms.image_slicer import KernelSlicer
import src.utils.factory as factory
import src.utils.pipeline_repository as pipeline_repository

FILE_NAME = Path(__file__).stem

def cmi_parse() -> tuple:
    parser = argparse.ArgumentParser(description= \
    "The preprocessing stage of the pipeline. This stage outputs preprocessed raw data, \
        and shards it to the preprocessed directory. Shards are tar archives")
    parser.add_argument("dataset", type=str, help="dataset package name")
    parser.add_argument("kernel_size", type=int, help="kernel size of the chunks cropper")
    parser.add_argument("overlap_perc", type=float, help="overlaping percentage between chunks")
    parser.add_argument("strategy", choices=["crop", "pad"], help="strategy for adjusting images when cropping yields non-aligned chunks")
    parser.add_argument("format", help="algorithm for image compression")
    parser.add_argument("--input", default="data/AerialImageDataset", help="input data directory")
    parser.add_argument("--output", default=f"{FILE_NAME}/output", help="output data directory")

    args = parser.parse_args()
    input_dir   = args.input
    output_dir  = args.output

    dataset = args.dataset
    kernel_size = args.kernel_size
    overlap_perc = args.overlap_perc
    strategy = args.strategy
    formats = args.format
    return dataset, kernel_size, overlap_perc, strategy, formats, Path(input_dir), Path(output_dir)

def process(dataset: Dataset, formats: str, output_dir: Path) -> None:
    img, mask = dataset[0]
    chunk_size = len(img)
    with tqdm(total=chunk_size*len(dataset)) as pbar:
        for i in range(len(dataset)):
            example_name = dataset.data[i].stem
            image_chunks, mask_chunks = dataset[i]
            for j in range(len(image_chunks)):
                img, mask = image_chunks[j], mask_chunks[j]
                chunk_index = i*len(image_chunks) + j
                image_path = Path(f"{output_dir}/img-{example_name}-{chunk_index}.{formats}") # TODO: Mozda bi bilo dobro da mijenjam naziv primjera
                mask_path  = Path(f"{output_dir}/mask-{example_name}-{chunk_index}.{formats}")
                pipeline_repository.push_images(output_dir, [img, mask], [image_path.name, mask_path.name])
                pbar.update()

if __name__ == "__main__":
    dataset_pckg, kernel_size, overlap_perc, strategy, formats, input_dir, output_dir = cmi_parse()
    slicer = KernelSlicer(kernel_size, overlap_perc, strategy=strategy)
    dataset = factory.get_object_from_standard_name(dataset_pckg)(root=input_dir, transforms=slicer)
    process(dataset, formats, output_dir)

