from ray.tune import schedulers
from .PopulationBasedTrainingScheduler import PopulationBasedTrainingScheduler

class SchedulerFactory:
    def create(self, name: str, **kwargs):
        # name = name.lower()
        # if    name == "pbt"      : return PopulationBasedTrainingScheduler(config)
        # else                     : raise NotImplementedError()

        # return self.scheduler = PopulationBasedTraining(
        #     time_attr             = 'training_iteration',
        #     # metric                = 'mean_accuracy',
        #     # mode                  = 'max',
        #     perturbation_interval = 5,
        #     hyperparam_mutations  = hyperparam_mutations
        # )

        return schedulers.SCHEDULER_IMPORT[name](**kwargs)
