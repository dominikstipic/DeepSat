from pathlib import Path

from . import shared_logic as shared_logic
import pipeline.evaluation as evaluation
import src.utils.factory as factory
import src.utils.pipeline_repository as pipeline_repository
import runners.trainer as trainer

FILE_NAME = Path(__file__).stem
MODEL_INPUT   = Path("trainer/output/model.pickle")
DATASET_INPUT  = Path("dataset_factory/output/test_db.pickle")
OUTPUT = Path(f"{FILE_NAME}/artifacts") 

def prepare_pip_arguments(config_args: dict):
    global INPUT, OUTPUT
    args = {}
    args["model"]  = pipeline_repository.get_pickle(MODEL_INPUT)
    args["device"] = config_args["device"]
    dataset = pipeline_repository.get_pickle(DATASET_INPUT)
    args["test_ld"] = factory.import_object(config_args["dataloader"], test_db=dataset)
    args["observers_dict"] = trainer.get_observers(config_args["observers"])
    args["output_dir"] = OUTPUT
    return args

if __name__ == "__main__":
    args = shared_logic.get_pipeline_stage_args(FILE_NAME)
    shared_logic.save_args(FILE_NAME)
    shared_logic.log_arguments(FILE_NAME, args)
    processed_args = prepare_pip_arguments(args)
    evaluation.process(**processed_args)
