from tqdm import tqdm
from pathlib import Path

from torch.utils.data import Dataset

import src.utils.pipeline_repository as pipeline_repository

def process(dataset: Dataset, format: str, output_dir: Path) -> None:
    img, mask = dataset[0]
    chunk_size = len(img)
    with tqdm(total=chunk_size*len(dataset)) as pbar:
        for i in range(len(dataset)):
            example_name = dataset.data[i].stem
            image_chunks, mask_chunks = dataset[i]
            for j in range(len(image_chunks)):
                img, mask = image_chunks[j], mask_chunks[j]
                chunk_index = i*len(image_chunks) + j
                image_path = Path(f"{output_dir}/img-{example_name}-{chunk_index}.{format}") # TODO: Mozda bi bilo dobro da mijenjam naziv primjera
                mask_path  = Path(f"{output_dir}/mask-{example_name}-{chunk_index}.{format}")
                pipeline_repository.push_images(output_dir, [img, mask], [image_path.name, mask_path.name])
                pbar.update()


