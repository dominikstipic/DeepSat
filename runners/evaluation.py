from pathlib import Path
import argparse

from . import shared_logic as shared_logic
import pipeline.evaluation as evaluation
import src.utils.factory as factory
import src.utils.pipeline_repository as pipeline_repository
import runners.trainer as trainer

FILE_NAME = Path(__file__).stem
MODEL_INPUT   = Path("trainer/output/model.pickle")
DATASET_INPUT  = Path("dataset_factory/output/test_db.pickle")
OUTPUT = Path(f"{FILE_NAME}/artifacts") 

def cmi_parse() -> dict:
    parser = argparse.ArgumentParser(description="Runner parser")
    parser.add_argument("--config", default="config.json", help="Configuration path")
    parser.add_argument("--model_input", default="trainer/output/model.pickle", help="Input directory")
    parser.add_argument("--dataset_input", default="dataset_factory/output/test_db.pickle", help="Input directory")
    parser.add_argument("--output", default=f"{FILE_NAME}/output", help="Output directory")
    args = vars(parser.parse_args())
    args = {k: Path(v) for k,v in args.items()}
    config_path = args["config"]
    args["config"] = shared_logic.get_pipeline_stage_args(config_path, FILE_NAME)
    return config_path, args

def prepare_pip_arguments(config: dict, dataset_input: Path, model_input: Path, output: Path):
    args = {}
    args["model"]  = pipeline_repository.get_pickle(model_input)
    args["device"] = config["device"]
    dataset = pipeline_repository.get_pickle(dataset_input)
    args["test_ld"] = factory.import_object(config["dataloader"], test_db=dataset)
    args["observers_dict"] = trainer.get_observers(config["observers"])
    args["output_dir"] = output
    return args

if __name__ == "__main__":
    config_path, args = cmi_parse()
    processed_args = prepare_pip_arguments(**args)
    shared_logic.prerun_routine(config_path, FILE_NAME)
    evaluation.process(**processed_args)
