from pathlib import Path
import argparse

from . import shared_logic as shared_logic
import pipeline.evaluation as evaluation
import src.utils.factory as factory
import src.utils.pipeline_repository as pipeline_repository
import runners.trainer as trainer
from src.transforms.transforms import Compose

FILE_NAME = Path(__file__).stem

def cmi_parse() -> dict:
    MODEL_INPUT   = Path("trainer/output")
    DATASET_INPUT  = Path("dataset_factory/output/test_db.pickle")
    OUTPUT = Path(f"{FILE_NAME}/output")
    model_input = pipeline_repository.get_path(MODEL_INPUT)
    dataset_input = pipeline_repository.get_path(DATASET_INPUT)
    output_dir = pipeline_repository.get_path(OUTPUT)

    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--model_input", default=model_input, help="Input directory")
    parser.add_argument("--dataset_input", default=dataset_input, help="Input directory")
    parser.add_argument("--output", default=output_dir, help="Output directory")
    args = vars(parser.parse_args())
    args = {k: Path(v) for k,v in args.items()}
    config_path = args["config"]
    args["config"] = shared_logic.get_pipeline_stage_args(config_path, FILE_NAME)
    return config_path, args

def get_model(model_input):
    model_data = pipeline_repository.get_objects(model_input)
    model, weights = model_data["model"], model_data["weights"]
    model.load_state_dict(weights)
    model.eval()
    return model

def get_postprocess(postprocess_list: list):
    postprocess_list = [factory.import_object(postprocess_item) for postprocess_item in postprocess_list]
    postprocess = Compose(postprocess_list)
    return postprocess

def prepare_pip_arguments(config: dict, dataset_input: Path, model_input: Path, output: Path):
    args = {}
    args["model"]  = get_model(model_input)
    args["device"] = config["device"]
    dataset = pipeline_repository.get_pickle(dataset_input)
    args["test_ld"] = factory.import_object(config["dataloader"], test_db=dataset)
    args["observers_dict"] = trainer.get_observers(config["observers"])
    args["postprocess"] = get_postprocess(config["postprocess"])
    args["output_dir"] = output
    return args

def process():
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    evaluation.process(**processed_args)

if __name__ == "__main__":
    process()
