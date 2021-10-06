from os import stat
from pathlib import Path
from PIL import Image
import io

import numpy as np
import matplotlib.pyplot as plt

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
    freq = [len(dataset)]
    results = dict(mean=means, std=stds, freq=freq)
    return results

def get_stats_dict(dataset_splits: dict):
    stats_dict = {}
    for split_name, dataset in dataset_splits.items():
        stats = dataset_statistics(dataset)
        stats_dict[split_name] = stats
    return stats_dict

def save_stat_plots(stats: dict, output_dir: Path):
    output_dir = pipeline_repository.create_dir_if_not_exist(output_dir)
    splits = []
    metrics = {k: [] for k in stats[list(stats.keys())[0]].keys()}
    for split, metrics_dict in stats.items():
        splits.append(split)
        for metric_key, metric_values in metrics_dict.items():
            metrics[metric_key].append(metric_values)
    start, spacing = 0, 1
    for k, (metric_name, metric_values) in enumerate(metrics.items()):
        for val in metric_values:
            end = start + len(val)
            keys = np.arange(start, end)
            plt.bar(keys, val, label=splits[k])
            start = end + spacing
        plt.legend()
        plt.savefig(output_dir / f"{metric_name}.png")  
        plt.clf()

#########################

def process(dataset_splits: dict, viz_samples: int, output: Path) -> None:
    example_artefacts = output / "examples"
    for split_name, dataset in dataset_splits.items():
        save_examples(dataset, viz_samples, example_artefacts / split_name)
    stats = get_stats_dict(dataset_splits)
    save_stat_plots(stats, output)
    pipeline_repository.push_json(output, "stats", stats)




