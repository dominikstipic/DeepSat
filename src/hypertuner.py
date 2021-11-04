from ray import tune

from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.bayesopt import BayesOptSearch

search_algs = {
    "basic": BasicVariantGenerator(),
    "bayes": BayesOptSearch()
}

class HyperTuner:
    def __init__(self, search_space, search_algorithm, resources_per_trial, num_samples):
        self.search_space = search_space
        self.search_algorithm = search_algorithm
        #self.search_scheduler = None
        self.active = False
        self.analysis = None
        self.resources_per_trial = resources_per_trial
        self.num_samples = num_samples
    
    def run(self, trainable):
        self.analysis = tune.run(trainable,
                                 config=self.search_space, 
                                 search_alg=self.search_algorithm,
                                 resources_per_trial=self.resources_per_trial,
                                 num_samples=self.num_samples,
                                 metric="performance",
                                 mode="max",
                                 verbose=0)
        df = self.analysis.results_df
        return df


