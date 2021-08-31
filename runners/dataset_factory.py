import os
from pathlib import Path

from . import shared_logic as shared_logic
from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory

def get_composite_transf(transformations: list) -> Compose: 
    transf_list = []
    for transf_dict in transformations: 
        transf = factory.import_object(transf_dict)
        transf_list.append(transf)
    return Compose(transf_list)

def get_train_test_transformations(trainsformation_dict: dict):
     train_transf, test_transf = trainsformation_dict["train"], trainsformation_dict["test"]
     train_transf, test_transf = get_composite_transf(train_transf), get_composite_transf(test_transf)
     return train_transf, test_transf

if __name__ == "__main__":
    file_name = Path(__file__).stem
    args = shared_logic.get_pipeline_stage_args(file_name)
    dataset_pckg = args['dataset']
    transformations = args["transformations"]
    test_ratio  = args["test_ratio"]
    valid_ratio = args["valid_ratio"]

    train_transf, test_transf = get_train_test_transformations(transformations)
    train_transf_pickle_path = pipeline_repository.push_pickled_obj(file_name, "objects/transformation", train_transf, "train_transf")
    test_transf_pickle_path = pipeline_repository.push_pickled_obj(file_name, "objects/transformation", test_transf, "test_transf")

    shared_logic.log_arguments(file_name, args)
    os.system(f"python -m pipeline.{file_name} {dataset_pckg} {str(train_transf_pickle_path)} {str(test_transf_pickle_path)} {test_ratio} {valid_ratio}")