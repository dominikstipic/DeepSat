from numpy import save
from tqdm import tqdm
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import src.utils.pipeline_repository as pipeline_repository
import src.utils.hashes as hashes

FILE_NAME = Path(__file__).stem

def cmi_parse() -> tuple:
    parser = argparse.ArgumentParser(description= \
    "Vizualizes samples and does statistics on data.")
    parser.add_argument("viz_samples", type=int, help="The number of samples to vizualize")
    parser.add_argument("--input", default=f"preprocess/output", help="input data directory")
    parser.add_argument("--output", default=f"{FILE_NAME}/artifacts", help="output data directory")
    args = vars(parser.parse_args())
    return args

def sample_images(images_root: Path, samples: int):
    extract_name = lambda p : f"{p.stem.split('-')[1]}-{p.stem.split('-')[2]}.{p.name.split('.')[1]}"
    to_full_path = lambda name: (images_root / f"img-{name}", images_root / f"mask-{name}")
    paths = list(set(map(lambda p: extract_name(p), images_root.iterdir())))
    paths = sorted(paths)
    string_data = " ".join(paths)
    generator = hashes.HashGenerator(string_data, scale=len(paths))
    samples = generator.sample(samples)
    sampled_paths = [paths[idx] for idx in samples]
    paths = list(map(lambda p : to_full_path(p), sampled_paths))
    return paths

def save_examples(examples: list, output_dir: Path, artifact_dir="examples"):
    for img_path, mask_path in examples:
        name = f"{img_path.stem.split('-')[1]}-{img_path.stem.split('-')[2]}{img_path.suffix}"
        img  = Image.open(str(img_path))
        mask = Image.open(str(mask_path))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(mask)
        pipeline_repository.create_dir_if_not_exist(output_dir / artifact_dir)
        output = pipeline_repository.get_path(output_dir / artifact_dir) / name
        plt.savefig(output)

def process(viz_samples: int, input: str, output: str) -> None:
    input = Path(pipeline_repository.get_path(input))
    output = Path(pipeline_repository.get_path(output))
    examples = sample_images(input, viz_samples)
    save_examples(examples, output)

if __name__ == "__main__":
    args = cmi_parse()
    process(**args)

