from pathlib import Path
from PIL import Image

from src.utils.common import merge_list_2d
import src.utils.hashes as hashes
from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository
import src.utils.common as common

from src.datasets.inria import Inria

def _classify_example(path: Path, train_ratio: float, valid_ratio: float, scale=100):
    digit = hashes.from_string(str(path), scale)
    train_ratio, valid_ratio = train_ratio*scale, valid_ratio*scale
    if digit >= 0 and digit < train_ratio: 
        return "train"
    elif digit >= train_ratio and digit < (valid_ratio + train_ratio):
        return "valid"
    else: 
        return "test" 

def _split_data(input_dir: Path, test_ratio: float, valid_ratio: float) -> tuple:
    train_ratio = 1 - test_ratio - valid_ratio
    splits = {}
    for path in input_dir.iterdir():
        example_split = _classify_example(path, train_ratio, valid_ratio)
        if example_split not in splits:
            splits[example_split] = []
        splits[example_split].append(str(path))
    return splits

def _save_splits_in_csv(datasets: dict, output_dir: Path, csv_name="splits.csv"):
    write_function = lambda split: lambda name: [Path(name).stem, split]
    for i, (split_name, dataset) in enumerate(datasets.items()):
        paths = dataset.get_paths()
        paths = list(map(lambda path: Path(path).stem, paths))
        to_append = i != 0
        pipeline_repository.push_csv(output_dir, \
                                     csv_name, \
                                     csv_header=["example", "split"], \
                                     data=paths, \
                                     write_function=write_function(split_name), \
                                     append=to_append)

def _sample_images_from_splits(paths: list, sample_num: int) -> tuple:
    if not len(paths) >= sample_num:
        return [], []
    paths = sorted(paths)
    string_data = " ".join(paths)
    generator = hashes.HashGenerator(string_data, len(paths))
    samples = generator.sample(sample_num)
    sampled_paths = [paths[idx] for idx in samples]

    imgs = [Image.open(img_path) for img_path in sampled_paths]
    get_mask_path = lambda str_path: Path(str_path).parent / f"mask-{Path(str_path).name[4:]}"
    masks = [Image.open(get_mask_path(img_path)) for img_path in sampled_paths]
    sampled_paths = [Path(path).name for path in sampled_paths]
    examples, example_names = list(zip(imgs, masks)), [[p, str(get_mask_path(p))] for p in sampled_paths]
    return examples, example_names

def _save_augmented_images(output_dir: Path, samples: int, datasets: dict, aug_dict: dict):
    for split_name, dataset in datasets.items():
        paths = dataset.get_paths()
        examples, example_names = _sample_images_from_splits(paths, samples)
        transf = aug_dict[split_name]

        transf_examples = [transf([img, mask]) for img, mask in examples]
        transf_names    = [[f"T-{img_name}", f"T-{mask_name}"] for img_name, mask_name in example_names]
        
        examples        = merge_list_2d(examples)
        example_names   = merge_list_2d(example_names)
        transf_examples = merge_list_2d(transf_examples)
        transf_names    = merge_list_2d(transf_names)

        multiply_points = lambda img: img.point(lambda pnx: 255*pnx) if img.mode == "L" else img
        examples = list(map(lambda img: multiply_points(img), examples))
        transf_examples = list(map(lambda img: multiply_points(img), transf_examples))

        examples = [common.h_concatenate_images(img, transf_img) for img, transf_img in list(zip(examples, transf_examples))]
        split_output_dir = pipeline_repository.create_dir_if_not_exist(output_dir / split_name)
        pipeline_repository.push_images(split_output_dir, examples, example_names)

# TODO - polymorphic call, don't ask for types
def _create_split_datasets(dataset, tensor_tf_dict: dict, aug_dict: dict, splits: dict):
    datasets = {}
    for split_name, split_paths in splits.items():
        dataset_copy = dataset.copy()
        split_tensor_tf, split_aug_tf = tensor_tf_dict[split_name], aug_dict[split_name]
        split_transform = Compose.from_composits(split_tensor_tf, split_aug_tf) 
        if type(dataset_copy) == Inria:
            imgs  = list(filter(lambda str_path: Path(str_path).stem.startswith("img"), split_paths))
            masks = list(filter(lambda str_path: Path(str_path).stem.startswith("mask"), split_paths))
            dataset_copy.data = imgs
            dataset_copy.labels = masks
        else:
            dataset_copy.tars = split_paths
        dataset_copy.transform  = split_transform
        datasets[split_name] = dataset_copy
    return datasets

def process(dataset, tensor_tf_dict: dict, aug_dict: dict, test_ratio: float, valid_ratio: float, viz_samples: int, input_dir: Path):
    pipeline_stage_name = Path(__file__).stem
    splits = _split_data(input_dir, test_ratio, valid_ratio)
    if not (tensor_tf_dict.keys() == aug_dict.keys() == splits.keys()):
        raise RuntimeError(f"Inconsisted configuration file: augmentation keys={list(aug_dict.keys())}, \
                             tensor transformation keys={list(tensor_tf_dict.keys())}, \
                             split keys={list(splits.keys())}")
    datasets = _create_split_datasets(dataset, tensor_tf_dict, aug_dict, splits)
    for split_name, split_dataset in datasets.items(): 
        pipeline_repository.push_pickled_obj(pipeline_stage_name, "output", split_dataset, f"{split_name}_db")
    
    # CSV file with splits
    csv_output_dir = Path(pipeline_stage_name) / "artifacts"
    _save_splits_in_csv(datasets, csv_output_dir)

    # Save augmented images and masks
    output_dir = Path(pipeline_stage_name) / "artifacts" / "transformations"
    _save_augmented_images(output_dir, viz_samples, datasets, aug_dict)
