from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import io

import torch

import src.utils.pipeline_repository as pipeline_repository

FILE_NAME = Path(__file__).stem

def sample_images(dataset, k: int, function=lambda x: x):
    examples = []
    for i, img in enumerate(dataset):
        if i >= k: break
        x = function(img)
        examples.append(x)
    return examples

#########################

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def create_subplots(img, mask): 
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img.permute(1,2,0))
    ax2.imshow(mask)
    pil_fig = fig2img(fig)
    plt.close(fig)
    return pil_fig

def get_subplots(dataset, viz_samples: int):
    examples = sample_images(dataset, viz_samples)
    subplots = list(map(lambda xs: create_subplots(*xs),examples))
    return subplots

def save_examples(dataset, viz_samples: int, path: Path):
    examples = sample_images(dataset, viz_samples)
    for idx, example in enumerate(examples):
        subplot = create_subplots(*example)
        pipeline_repository.push_images(path, images=[subplot], names=[idx])
        
#########################

def dataset_statistics(dataset):
    calc_mean = lambda xs: xs[0].view(3,-1).mean(1)
    calc_std  = lambda xs: xs[0].view(3,-1).std(1)
    aggregate = lambda xs: torch.stack(xs).mean(0).tolist()
    means = sample_images(dataset, len(dataset), calc_mean)
    means = aggregate(means)
    stds = sample_images(dataset, len(dataset), calc_std)
    stds = aggregate(stds)
    results = dict(mean=means, std=stds)
    return results

def get_stats_dict(train_db, test_db, valid_db):
    train_stats = dataset_statistics(train_db)
    valid_stats = dataset_statistics(valid_db)
    test_stats  = dataset_statistics(test_db)
    stats = dict(train=train_stats, valid=valid_stats, test=test_stats)
    return stats

#########################

def process(train_db, test_db, valid_db, viz_samples: int, output: Path) -> None:
    example_artefacts = output / "examples"
    save_examples(train_db, viz_samples, example_artefacts / "train")
    save_examples(test_db,  viz_samples, example_artefacts / "test")
    save_examples(valid_db, viz_samples, example_artefacts / "valid")

    stats = get_stats_dict(train_db, test_db, valid_db)
    pipeline_repository.push_json(output, "stats", stats)




