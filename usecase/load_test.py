from ray import tune
import json

path = "/home/tomoya-y/ray_results/tune_WWW/experiment_state-2022-07-28_18-55-10.json"
# with open(path, "r") as f:
#     d = json.loads(f)


a = tune.ExperimentAnalysis(
    experiment_checkpoint_path=path
)

a = 4