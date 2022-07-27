import optuna


def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=30, interval_steps=10
    ))
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}

