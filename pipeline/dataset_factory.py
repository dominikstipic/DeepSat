import argparse
from pathlib import Path
from src.utils import factory
from PIL import Image

from src.utils.common import merge_list_2d, unpack_tar_archive_for_paths
import src.utils.hashes as hashes
from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
from src.utils.common import load_pickle

def cmi_parse() -> tuple:
    parser = argparse.ArgumentParser(description= \
   "The factory for the dataset pickle object. This stage applies transformations to the data, \
    splits the data to the train and valid parts and constructs DataLoaders. ")
    parser.add_argument("dataset", help="dataset package from src.datasets")
    parser.add_argument("train_transforms", help="path to pickle serialized train transformation objects")
    parser.add_argument("test_transforms", help="path to pickle serialized test transformation objects")
    parser.add_argument("test_ratio", type=float, help="ratio of test data to the all data")
    parser.add_argument("valid_ratio", type=float, help="ratio of valid data to the train data")
    parser.add_argument("--input", default="repository/sharding/output", help="input data directory")
    parser.add_argument("--samples", default=3, help="number of samples to save as artifacts. Samples are transformed images from different splits")
    args = parser.parse_args()

    dataset = args.dataset 
    train_transforms = args.train_transforms 
    test_transforms = args.test_transforms 
    test_ratio = args.test_ratio
    valid_ratio = args.valid_ratio
    samples = args.samples
    input_dir = args.input
    return dataset, Path(train_transforms), Path(test_transforms), test_ratio, valid_ratio, samples, Path(input_dir)

def classify_example(path: Path, train_ratio: float, valid_ratio: float, scale=100):
    digit = hashes.from_string(str(path), scale)
    train_ratio, valid_ratio = train_ratio*scale, valid_ratio*scale
    if digit >= 0 and digit < train_ratio: 
        return "train"
    elif digit >= train_ratio and digit < (valid_ratio + train_ratio):
        return "valid"
    else: 
        return "test" 

def split_data(input_dir: Path, test_ratio: float, valid_ratio: float) -> tuple:
    train_ratio = 1 - test_ratio - valid_ratio
    train, valid, test = [], [], []
    for path in input_dir.iterdir():
        example_split = classify_example(path, train_ratio, valid_ratio)
        if example_split == "train": train.append(str(path))
        elif example_split == "valid": valid.append(str(path))
        else: test.append(str(path))
    return train, valid, test

def sample_images_from_splits(shard_paths: list, sample_num: int) -> tuple:
    assert len(shard_paths) >= sample_num, "The number of samples must be larger than sumpling number"
    shard_paths = sorted(shard_paths)
    string_data = " ".join(shard_paths)
    generator = hashes.HashGenerator(string_data, len(shard_paths))
    samples = generator.sample(sample_num)
    sampled_paths = [shard_paths[idx] for idx in samples]

    imgs = [Image.open(img_path) for img_path in sampled_paths]
    get_mask_path = lambda str_path: Path(str_path).parent / f"mask-{Path(str_path).name[4:]}"
    masks = [Image.open(get_mask_path(img_path)) for img_path in sampled_paths]
    sampled_paths = [Path(path).name for path in sampled_paths]
    examples, example_names = list(zip(imgs, masks)), [[p, str(get_mask_path(p))] for p in sampled_paths]
    return examples, example_names

def save_splits_in_csv(pipeline_stage_name: str, train_shard_paths: list, valid_shard_paths: list, test_shard_paths: list):
    write_function = lambda split: lambda name: [Path(name).stem, split]
    
    train_paths = merge_list_2d(list(map(lambda shard: unpack_tar_archive_for_paths(shard), train_shard_paths)))
    valid_paths = merge_list_2d(list(map(lambda shard: unpack_tar_archive_for_paths(shard), valid_shard_paths)))
    test_paths  = merge_list_2d(list(map(lambda shard: unpack_tar_archive_for_paths(shard), test_shard_paths)))

    train_paths = list(filter(lambda path: Path(path).stem.startswith("img"), train_paths))
    valid_paths = list(filter(lambda path: Path(path).stem.startswith("img"), valid_paths))
    test_paths = list(filter(lambda path: Path(path).stem.startswith("img"), test_paths))

    pipeline_repository.push_csv(pipeline_stage_name, "splits.csv", csv_header=["example", "split"], data=train_paths, default_dir="artifacts", write_function=write_function("train"), append=False)
    pipeline_repository.push_csv(pipeline_stage_name, "splits.csv", csv_header=["example", "split"], data=valid_paths, default_dir="artifacts", write_function=write_function("valid"), append=True)
    pipeline_repository.push_csv(pipeline_stage_name, "splits.csv", csv_header=["example", "split"], data=test_paths,  default_dir="artifacts", write_function=write_function("test"),  append=True)
    return train_paths, valid_paths, test_paths

def save_transformed_images(pipeline_stage_name: str, samples: int, train_paths: list, valid_paths: list, test_paths: list, train_transf: Compose, test_transf: Compose):
    train_examples, train_names = sample_images_from_splits(train_paths, samples)
    valid_examples, valid_names = sample_images_from_splits(valid_paths, samples)
    test_examples, test_names   = sample_images_from_splits(test_paths, samples)

    get_transf_images = lambda examples, transf: [transf([img, mask]) for img, mask in examples]
    get_transf_names  = lambda names : [[f"T-{img_name}", f"T-{mask_name}"] for img_name, mask_name in  names]
    train_transf_images, valid_transf_images, test_transf_images = get_transf_images(train_examples, train_transf), get_transf_images(valid_examples, train_transf), get_transf_images(test_examples, test_transf)
    train_transf_names, valid_transf_names, test_transf_names = get_transf_names(train_names), get_transf_names(valid_names), get_transf_names(test_names)

    train_transf_images, valid_transf_images, test_transf_images = merge_list_2d(train_transf_images), merge_list_2d(valid_transf_images), merge_list_2d(test_transf_images)
    train_transf_names, valid_transf_names, test_transf_names = merge_list_2d(train_transf_names), merge_list_2d(valid_transf_names), merge_list_2d(test_transf_names)
    train_examples, valid_examples, test_examples = merge_list_2d(train_examples), merge_list_2d(valid_examples), merge_list_2d(test_examples)
    train_names, valid_names, test_names = merge_list_2d(train_names), merge_list_2d(valid_names), merge_list_2d(test_names)
    
    multiply_points = lambda img: img.point(lambda pnx: 255*pnx) if img.mode == "L" else img
    train_examples = list(map(lambda img: multiply_points(img), train_examples))
    valid_examples = list(map(lambda img: multiply_points(img), valid_examples))
    test_examples  = list(map(lambda img: multiply_points(img), test_examples))

    train_transf_images = list(map(lambda img: multiply_points(img), train_transf_images))
    valid_transf_images = list(map(lambda img: multiply_points(img), valid_transf_images))
    test_transf_images  = list(map(lambda img: multiply_points(img), test_transf_images))

    root_dir = Path(pipeline_stage_name) / "artifacts" / "transformations"
    _ = pipeline_repository.create_dir(root_dir)
    train_dir = pipeline_repository.create_dir(root_dir / "train")
    valid_dir = pipeline_repository.create_dir(root_dir / "valid")
    test_dir = pipeline_repository.create_dir(root_dir / "test")

    pipeline_repository.push_images(train_dir, train_examples, train_names)
    pipeline_repository.push_images(valid_dir, valid_examples, valid_names)
    pipeline_repository.push_images(test_dir,  test_examples,  test_names)

    pipeline_repository.push_images(train_dir, train_transf_images, train_transf_names)
    pipeline_repository.push_images(valid_dir, valid_transf_images, valid_transf_names)
    pipeline_repository.push_images(test_dir, test_transf_images, test_transf_names)

def save_datasets(pipeline_stage_name: str, dataset, test_transf: Compose, train_shard_paths: list, valid_shard_paths: list, test_shard_paths: list):
    train_db, valid_db, test_db = dataset.copy(), dataset.copy(), dataset.copy()
    train_db.tars = train_shard_paths
    valid_db.tars = valid_shard_paths
    test_db.tars  = test_shard_paths
    test_db.transform = test_transf

    pipeline_repository.push_pickled_obj(pipeline_stage_name, "output/datasets", train_db, "train_db")
    pipeline_repository.push_pickled_obj(pipeline_stage_name, "output/datasets", valid_db, "valid_db")
    pipeline_repository.push_pickled_obj(pipeline_stage_name, "output/datasets", test_db, "test_db")

def process(dataset_pckg: str, train_transf: Compose, test_transf: Compose, test_ratio: float, valid_ratio: float, samples: int, input_dir: Path):
    pipeline_stage_name = Path(__file__).stem
    train_shard_paths, valid_shard_paths, test_shard_paths = split_data(input_dir, test_ratio, valid_ratio)
    dataset = factory.get_object_from_standard_name(dataset_pckg)(input_dir, train_transf)

    save_datasets(pipeline_stage_name, dataset, test_transf, train_shard_paths, valid_shard_paths, test_shard_paths)
    ####### artifacts for UI #######
    # CSV file with splits
    train_paths, valid_paths, test_paths = save_splits_in_csv(pipeline_stage_name, train_shard_paths, valid_shard_paths, test_shard_paths)
    # Transformed images and masks
    save_transformed_images(pipeline_stage_name, samples, train_paths, valid_paths, test_paths, train_transf, test_transf)

if __name__ == "__main__":
    dataset_pckg, train_transforms_path, test_transforms_path, test_ratio, valid_ratio, samples, input_dir = cmi_parse()

    train_transf = load_pickle(train_transforms_path)
    test_transf  = load_pickle(test_transforms_path)

    process(dataset_pckg, train_transf, test_transf, test_ratio, valid_ratio, samples, Path(input_dir))