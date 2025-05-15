import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

import os
from datetime import datetime
from functools import partial

import optuna.trial
import wandb
from torch.nn import Sequential

from tqdm import tqdm
from xgboost.callback import TrainingCallback
from xgboost import XGBRegressor, Booster
from wandb.integration.xgboost import WandbCallback

from src.models.ml.dataset import MLDataset
from src.models.ml.transforms import AddMonth, AddHour, AddWeekDay, AddWeekNumber, Normalize, compact_data
from src.models.ml.metrics import plot_predictions, compute_metrics_keras


START_TIME = datetime.now().isoformat()
PROJECT_NAME="metraq-aq-testing-xgboost"
GROUP_NAME=f"{START_TIME}-xgboost"
# code_test_size=500


class PruningCallback(TrainingCallback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def after_iteration(self, model, epoch, evals_log):
        # Report intermediate metrics for pruning
        test_mae = evals_log.get('validation_1', {}).get('mae', [None])[-1]
        if test_mae is not None:
            self.trial.report(test_mae, epoch)

        # If the pruner decides that the trial should be pruned
        if self.trial.should_prune():
            print(f"Trial was pruned at epoch: {epoch}")
            raise optuna.TrialPruned()

        return False  # Continue training


class ProgressMonitor(TrainingCallback):
    def __init__(self, pbar: tqdm = None):
        super().__init__()
        self.pbar = pbar

    def before_iteration(self, model, epoch, evals_log):
        self.pbar.set_description(f"Processing estimator {epoch}")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)

        # Access training and test validation logs
        train_mae = evals_log.get('validation_0', {}).get('mae', [None])[-1]
        test_mae = evals_log.get('validation_1', {}).get('mae', [None])[-1]

        # Display both train and test MAE in the progress bar
        self.pbar.set_postfix(train_mae=f"{train_mae:.5f}" if train_mae is not None else "N/A",
                              test_mae=f"{test_mae:.5f}" if test_mae is not None else "N/A")

        return False  # Continue training


class CustomWandbCallback(WandbCallback):
    def __init__(self, names_mappings: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.names_mappings = names_mappings

    def after_iteration(self, model: Booster, epoch: int, evals_log: dict) -> bool:
        """Run after each iteration. Return True when training should stop."""
        # Log metrics
        for data, metric in evals_log.items():
            if self.names_mappings is not None:
                data = data if data not in self.names_mappings.keys() else self.names_mappings[data]

            for metric_name, log in metric.items():

                if self.define_metric:
                    self._define_metric(data, metric_name)
                    wandb.log({f"{data}-{metric_name}": log[-1]}, commit=False)
                else:
                    wandb.log({f"{data}-{metric_name}": log[-1]}, commit=False)

        wandb.log({"epoch": epoch})
        self.define_metric = False
        return False

def get_dataset(data_file: str, x_steps: int, y_steps: int, train_years: list, test_years: list, x_mag: list,
                y_mag: list, normalize: bool = False):
    # TODO:
    #    Validate years (train and test years can't contain same values)
    #    Parametrize augmentations, ¿maybe through magnitudes?

    transforms = Sequential(
        AddHour(),
        AddWeekDay(),
        AddWeekNumber()
    )

    if normalize:
        transforms = Sequential(Normalize(), *transforms)

    train_dataset = None
    if train_years is not None and len(train_years) > 0:
        train_dataset = MLDataset(data_file=data_file, x_steps=x_steps, y_steps=y_steps, years=train_years,
                            x_magnitudes=x_mag, y_magnitudes=y_mag, transform=transforms)
    test_dataset = None
    if test_years is not None and len(test_years) > 0:
        test_dataset = MLDataset(data_file=data_file, x_steps=x_steps, y_steps=y_steps, years=test_years,
                            x_magnitudes=x_mag, y_magnitudes=y_mag, transform=transforms)

    return train_dataset, test_dataset

def get_common_sensors(train_set, test_set):
    if train_set is None or train_set.sensors is None:
        if test_set is None or test_set.sensors is None:
            return []
        return test_set.sensors
    return list(set(train_set.sensors if train_set is not None else []) &
         set(test_set.sensors if test_set is not None else []))

def get_data(data_file: str, x_steps: int, y_steps: int, train_years: list, test_years: list, x_mag: list, y_mag: list,
             normalize: bool = False, sensors: list = None):
    train_set, test_set = get_dataset(data_file=data_file, x_steps=x_steps, y_steps=y_steps, train_years=train_years,
                          test_years=test_years, x_mag=x_mag, y_mag=y_mag, normalize=normalize)

    common_sensors = get_common_sensors(train_set, test_set) if sensors is None else sensors

    if train_years is not None and len(train_years) > 0:
        print("Preprocessing train data", flush=True)
        train_x, train_y, train_meta = compact_data(train_set, common_sensors)
        train_meta['sensors'] = common_sensors
    else:
        train_x = train_y = train_meta = None

    if test_years is not None and len(test_years) > 0:
        print("Preprocessing test data", flush=True)
        test_x, test_y, test_meta = compact_data(test_set, common_sensors)
        test_meta['sensors'] = common_sensors
    else:
        test_x = test_y = test_meta = None

    return train_x, train_y, train_meta, test_x, test_y, test_meta


def train_model(params: dict, data: dict, trial: optuna.trial.Trial = None):
    print("Training model...", flush=True)

    if data is None:
        (train_x, train_y, train_meta, test_x, test_y, test_meta) = get_data(data_file=params['data_file'],
                                               x_steps=params['x_steps'],
                                               y_steps=params['y_steps'],
                                               train_years=params['train_years'],
                                               test_years=params['eval_years'],
                                               x_mag=params['x_magnitudes'],
                                               y_mag=params['y_magnitudes'],
                                               normalize=params.get('normalize', False))
    else:
        train_x = data.get('train_x', None)
        train_y = data.get('train_y', None)
        train_meta = data.get('train_meta', None)
        test_x = data.get('test_x', None)
        test_y = data.get('test_y', None)
        test_meta = data.get('test_meta', None)


    wandb.init(project=PROJECT_NAME,
               group=GROUP_NAME,
               dir=os.path.join(params.get('output_folder', 'wandb'), PROJECT_NAME, GROUP_NAME),
               name=f"trial_{trial.number}" if trial else None)
    params['model_name'] = XGBRegressor.__class__.__name__
    params['sensors'] = train_meta['sensors']
    params['stats'] = train_meta['stats']
    wandb.config.update(params)

    eval_set = [(train_x, train_y), (test_x, test_y)] if test_x is not None and test_y is not None else None

    callbacks = [CustomWandbCallback(log_model=True,
                                     names_mappings={'validation_0': 'train', 'validation_1': 'test'}),
                 ProgressMonitor(pbar=tqdm(total=params['estimators']))]

    if trial is not None:
        callbacks.append(PruningCallback(trial))

    model = XGBRegressor(n_estimators=params['estimators'],
                         max_depth=params['max_depth'],
                         learning_rate=params['learning_rate'],
                         objective=params['objective'],
                         eval_metric=["mae", "rmse"],
                         gamma=0,
                         device="gpu",
                         callbacks=callbacks
                         )

    try:
        model.fit(train_x,
                  train_y,
                  eval_set=eval_set,
                  verbose=False)
    except optuna.TrialPruned:
        print("Trial was pruned.")
        wandb.log({'status': 'pruned'})
        wandb.finish()
        return None

    wandb.log({'status': 'finished'})
    wandb.finish()
    print("done", flush=True)

    final_test_mae = model.evals_result().get('validation_1', {}).get('mae', [-1])[-1] if eval_set else 0.0
    return final_test_mae


def eval_model(params):
    run = wandb.init(project=PROJECT_NAME)
    artifact = run.use_artifact(params['wandb_model'], type='model')
    artifact_dir = artifact.download()
    producer_run = artifact.logged_by()
    run_config = producer_run.config
    run_summary = producer_run.summary

    run_config.update(params)

    model_file = os.path.join(artifact_dir, artifact.name.split(':')[0])
    model = XGBRegressor()
    model.load_model(model_file)

    normalize = run_config.get('normalize', False)
    print(f"Evaluating model {producer_run.name} for years {run_config['eval_years']} "
          f"{'with' if normalize else 'without'} normalization ...", flush=True)

    _, _, _, test_x, test_y, test_meta = get_data(data_file=run_config['data_file'],
                                           x_steps=run_config['x_steps'],
                                           y_steps=run_config['y_steps'],
                                           train_years=[],
                                           test_years=run_config['eval_years'],
                                           x_mag=run_config['x_magnitudes'],
                                           y_mag=run_config['y_magnitudes'],
                                           normalize=normalize,
                                           sensors=run_config.get('sensors', None))

    predictions = model.predict(test_x)
    metrics = compute_metrics_keras(y_hat=predictions, y=test_y, metadata=test_meta, normalize=run_config.get('normalize', False))

    if params.get('plot_output_folder', None) is not None:
        plot_predictions(y_hat=predictions, y=test_y, metadata=test_meta, output_folder=params.get('plot_output_folder', None),
                         normalize=normalize)

    print(f"Evaluation metrics: {metrics}")

def params_to_optuna_categories(params: dict, keys: list):
    for key in keys:
        if key in params and not isinstance(params[key], list):
            params[key] = [params[key]]
    return params

def optuna_train_trial(trial: optuna.trial.Trial, params: dict, data: dict):
    params = params.copy()

    estimators = trial.suggest_categorical('estimators', choices=params['estimators'])
    max_depth = trial.suggest_categorical('max_depth', choices=params['max_depth'])
    learning_rate = trial.suggest_categorical('learning_rate', choices=params['learning_rate'])
    objective = trial.suggest_categorical('objetive', choices=params['objective'])
    x_magnitudes = trial.suggest_categorical('x_magnitudes', choices=params['x_magnitudes'])

    params.update({
        'estimators': estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'objective': objective,
        'x_magnitudes': x_magnitudes
    })

    return train_model(params=params, data=data)


def optuna_train(params):
    params = params_to_optuna_categories(params, ['estimators', 'max_depth', 'learning_rate', 'objective',
                                                  'x_magnitudes'])

    search_space = {x: params.get(x) for x in ['estimators', 'max_depth', 'learning_rate', 'objective',
                                                  'x_magnitudes']}

    # TODO: chapuza, revisar
    if 'x_magnitudes' not in search_space:
        (train_x, train_y, train_meta, test_x, test_y, test_meta) = get_data(data_file=params['data_file'],
                                               x_steps=params['x_steps'],
                                               y_steps=params['y_steps'],
                                               train_years=params['train_years'],
                                               test_years=params['eval_years'],
                                               x_mag=params['x_magnitudes'],
                                               y_mag=params['y_magnitudes'])

        data = {
            'train_x': train_x,
            'train_y': train_y,
            'train_meta': train_meta,
            'test_x': test_x,
            'test_y': test_y,
            'test_meta': test_meta
        }
    else:
        data = None

    optuna_train_with_params = partial(optuna_train_trial, params=params, data=data)

    # study = optuna.create_study(direction="minimize", study_name=PROJECT_NAME, pruner=optuna.pruners.MedianPruner())
    study = optuna.create_study(direction="minimize", study_name=PROJECT_NAME, pruner=optuna.pruners.MedianPruner(),
                                sampler=optuna.samplers.GridSampler(search_space=search_space))
    study.optimize(optuna_train_with_params, n_trials=100)


if __name__=="__main__":
    raise RuntimeError("This is a library, can't be launched as a script")