from pathlib import Path

from ray import tune
from ray.tune import Callback
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.bayesopt import BayesOptSearch

import src.utils.pipeline_repository as pipeline_repository


search_algs = {
    "basic": BasicVariantGenerator(),
    "bayes": BayesOptSearch()
}

class AfterTrailCallback(Callback):
    def __init__(self, path_dir: Path):
        pipeline_repository.create_dir_if_not_exist(path_dir)
        self.path = path_dir / "perf.txt"

    def update(self, config_str: str, perf: int):
        flag = "w"
        if self.path.exists(): flag = "a"
        with open(str(self.path), flag) as fp:
            line = f"{perf}---{config_str}"
            fp.write(line + "\n")

    def on_trial_result(self, trial, result, **kwargs):
        config_str = str(trial.config).replace("{","").replace("}","")
        perf = result["performance"]
        self.update(config_str, perf)
        print(f"{perf}---{config_str}")

class HyperTuner:
    path = pipeline_repository.get_path("trainer/artifacts")

    def __init__(self, search_space, search_algorithm, resources_per_trial, num_samples, iterations):
        self.search_space = search_space
        self.search_algorithm = search_algorithm
        #self.search_scheduler = None
        self.active = False
        self.analysis = None
        self.resources_per_trial = resources_per_trial
        self.num_samples = num_samples
        self.iterations = iterations
    
    def run(self, trainable):
        callback = AfterTrailCallback(self.path)
        self.analysis = tune.run(trainable,
                                 config=self.search_space, 
                                 search_alg=self.search_algorithm,
                                 resources_per_trial=self.resources_per_trial,
                                 num_samples=self.num_samples,
                                 metric="performance",
                                 mode="max",
                                 log_to_file=True,
                                 verbose=0, 
                                 callbacks=[callback])
        df = self.analysis.results_df
        return df


