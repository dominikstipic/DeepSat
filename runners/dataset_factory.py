from pathlib import Path
from runners.sharding import FILE_NAME

from . import shared_logic as shared_logic
from src.transforms.transforms import Compose
import src.utils.pipeline_repository as pipeline_repository
import src.utils.factory as factory
import pipeline.dataset_factory as dataset_factory

FILE_NAME = Path(__file__).stem
INPUT  = Path("sharding/output") # TODO: What if I change sharding name

def _get_composite_transf(transformations: list) -> Compose: 
    transf_list = []
    for transf_dict in transformations: 
        transf = factory.import_object(transf_dict)
        transf_list.append(transf)
    return Compose(transf_list)

def _get_train_test_transformations(trainsformation_dict: dict) -> tuple:
     train_transf, test_transf = trainsformation_dict["train"], trainsformation_dict["test"]
     train_transf, test_transf = _get_composite_transf(train_transf), _get_composite_transf(test_transf)
     return train_transf, test_transf

def prepare_pip_arguments(config_args: dict) -> dict:
    global INPUT, OUTPUT
    input_dir = pipeline_repository.get_path(INPUT)
    train_transf, test_transf = _get_train_test_transformations(config_args["transformations"])
    test_ratio, valid_ratio = config_args["test_ratio"], config_args["valid_ratio"]
    dataset = factory.get_object_from_standard_name(config_args["dataset"])(input_dir, train_transf)
    viz_samples = config_args["viz_samples"]
    args = locals()
    del args["config_args"]
    return args

if __name__ == "__main__":
    args = shared_logic.get_pipeline_stage_args(FILE_NAME)
    shared_logic.log_arguments(FILE_NAME, args)
    processed_args = prepare_pip_arguments(args)
    dataset_factory.process(**processed_args)