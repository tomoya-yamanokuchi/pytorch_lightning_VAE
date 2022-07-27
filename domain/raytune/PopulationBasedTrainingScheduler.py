import random
from ray.tune.schedulers.pbt import PopulationBasedTraining


class PopulationBasedTrainingScheduler:
    def __init__(self, hyperparam_mutations):
        self.scheduler = PopulationBasedTraining(
            time_attr             = 'training_iteration',
            # metric                = 'mean_accuracy',
            # mode                  = 'max',
            perturbation_interval = 5,
            hyperparam_mutations  = hyperparam_mutations
    )