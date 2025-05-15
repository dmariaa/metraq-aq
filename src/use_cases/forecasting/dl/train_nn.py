import os
from datetime import datetime
from functools import partial

import optuna.trial
import torch
import wandb
from torch.nn import Sequential
from torch.utils.data import DataLoader

from src.models.dl.dataset import MLDataset
from src.models.dl.nn import MLPRegression
from src.models.dl.torch_utils import train, get_loss, get_opt, inference
from src.models.dl.transforms import AddHour, AddWeekDay, Normalize
from src.models.tools.wandb import save_model_to_wandb

START_TIME = datetime.now().isoformat()
PROJECT_NAME="metraq-aq-deeplearning"
GROUP_NAME=f"{START_TIME}-nn"

def collate_fn_to_tuple(batch, common_sensors):
    x_tensors = []
    y_tensors = []
    for item in batch:
        df_x = item['x']
        df_x = df_x[df_x['sensor_id'].isin(common_sensors)]

        # filtered_x = df_x.drop(columns=['magnitude_id', 'sensor_id', 'entry_date'], errors='ignore')
        filtered_x = df_x.iloc[:, 2:]
        x_tensors.append(torch.tensor(filtered_x.values, dtype=torch.float32))

        df_y = item['y']
        df_y = df_y[df_y['sensor_id'].isin(common_sensors)]

        # filtered_y = df_y.drop(columns=['magnitude_id', 'sensor_id', 'entry_date'], errors='ignore')
        filtered_y = df_y.iloc[:, 2:]
        y_tensors.append(torch.tensor(filtered_y.values, dtype=torch.float32))

    return torch.stack(x_tensors), torch.stack(y_tensors)


def get_dataset(data_file: str, x_steps: int, y_steps: int, train_years: list, test_years: list, x_mag: list,
                y_mag: list, sensors: list = None):
    # TODO:
    #    Validate years (train and test years can't contain same values)
    #    Parametrize augmentations, ¿maybe through magnitudes?

    transforms = Sequential(
        Normalize(),
        AddHour(),
        AddWeekDay(),
        # AddWeekNumber(),
        # AddMonth(),
        # ToKeras()
    )

    if train_years is not None and len(train_years) > 0:
        train_dataset = MLDataset(data_file=data_file, x_steps=x_steps, y_steps=y_steps, years=train_years,
                            x_magnitudes=x_mag, y_magnitudes=y_mag, transform=transforms)
    else:
        train_dataset = None

    if test_years is not None and len(test_years) > 0:
        test_dataset = MLDataset(data_file=data_file, x_steps=x_steps, y_steps=y_steps, years=test_years,
                        x_magnitudes=x_mag, y_magnitudes=y_mag, transform=transforms)
    else:
        test_dataset = None

    if sensors is None:
        common_sensors: list = list(set([] if train_dataset is None else train_dataset.sensors) &
                                set([] if test_dataset is None else test_dataset.sensors))
    else:
        common_sensors = sensors

    return train_dataset, test_dataset, transforms, common_sensors


def train_model(params: dict, data: dict, trial: optuna.trial.Trial = None):
    print("Training model...", flush=True)

    train_dataset, test_dataset, transforms, common_sensors = get_dataset(data_file=params['data_file'],
                                                                          x_steps=params['x_steps'],
                                                                          y_steps=params['y_steps'],
                                                                          train_years=params['train_years'],
                                                                          test_years=params['eval_years'],
                                                                          x_mag=params['x_magnitudes'],
                                                                          y_mag=params['y_magnitudes'])

    variable = partial(collate_fn_to_tuple, common_sensors=common_sensors)

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=variable)
    valid_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=variable)

    n_x_magnitudes: int = len(train_dataset[0]['x_magnitudes'])
    n_y_magnitudes: int = len(train_dataset[0]['y_magnitudes'])
    input_dim_nn = n_x_magnitudes * params['x_steps'] * (len(common_sensors))
    output_dim_nn = n_y_magnitudes * params['y_steps'] * (len(common_sensors))

    wandb.init(project=PROJECT_NAME,
               group=GROUP_NAME,
               dir=os.path.join(params.get('output_folder', 'wandb'), PROJECT_NAME, GROUP_NAME),
               name=f"trial_{trial.number}" if trial else None)
    params['model_name'] = "Neural network"
    params['sensors'] = common_sensors
    params['input_dim'] = input_dim_nn
    params['output_dim'] = output_dim_nn
    params['stats'] = train_dataset.stats
    wandb.config.update(params)

    model = MLPRegression(input_dim=input_dim_nn, hidden_dim=512, output_dim=output_dim_nn).to(params['device'])

    try:
        criterion = get_loss(params['loss'])
        optimizer = get_opt(model=model, opt=params['optimizer'], lr=float(params['learning_rate']))

        train_loss, valid_loss, init_time, end_time = train(epochs=params['n_epochs'],
                                                            model=model,
                                                            train_dl=train_dataloader,
                                                            valid_dl=valid_dataloader,
                                                            loss_fn=criterion,
                                                            device=params['device'],
                                                            optimizer=optimizer,
                                                            wandb_log=wandb.log)

    except optuna.TrialPruned:
        print("Trial was pruned.")
        wandb.log({'status': 'pruned'})
        wandb.finish()
        return None

    wandb.log({'status': 'finished'})
    save_model_to_wandb(model=model, model_name=model.__class__.__name__, metadata=params, file_format="pt")

    model_name = model.__class__.__name__
    model_file_name = f"{model_name}.pt"
    torch.save(model.state_dict(), model_file_name)
    artifact = wandb.Artifact(
        name=model_name,
        type='model',
        description="MLPRegression neural network model",
        metadata=params
    )
    artifact.add_file(model_file_name)
    wandb.run.log_artifact(artifact)

    wandb.finish()
    print("done", flush=True)

    return valid_loss


def eval_model(params):
    run = wandb.init(project=PROJECT_NAME)
    artifact = run.use_artifact(params['wandb_model'], type='model')
    artifact_dir = artifact.download()
    producer_run = artifact.logged_by()
    run_config = producer_run.config
    run_summary = producer_run.summary

    run_config.update(params)

    _, test_dataset, transforms, common_sensors = get_dataset(data_file=run_config['data_file'],
                                                              x_steps=run_config['x_steps'],
                                                              y_steps=run_config['y_steps'],
                                                              train_years=[],
                                                              test_years=run_config['eval_years'],
                                                              x_mag=run_config['x_magnitudes'],
                                                              y_mag=run_config['y_magnitudes'],
                                                              sensors=run_config['sensors'])
    variable = partial(collate_fn_to_tuple, common_sensors=common_sensors)
    test_dataloader = DataLoader(test_dataset, batch_size=run_config['batch_size'], shuffle=False, collate_fn=variable)

    input_dim_nn = run_config['input_dim']
    output_dim_nn = run_config['output_dim']
    model = MLPRegression(input_dim=input_dim_nn, hidden_dim=512, output_dim=output_dim_nn).to(run_config['device'])

    model_file_name = f"{model.__class__.__name__}.pt"
    model_file = os.path.join(artifact_dir, model_file_name)
    model.load_state_dict(torch.load(model_file))

    print(f"Evaluating model {run_config['model_name']} for years {run_config['eval_years']} ...", flush=True)

    # stats can be obtained from configuration
    magnitude_id = run_config['y_magnitudes'][0]
    min = run_config['stats'][str(magnitude_id)]['min']
    max = run_config['stats'][str(magnitude_id)]['max']

    # or from the dataset itself
    # stats = test_dataset.stats
    # min = stats[magnitude_id]['min']
    # max = stats[magnitude_id]['max']

    test_mae, test_mse, test_mae_norm, test_mse_norm = inference(model=model, test_dl=test_dataloader, device=run_config['device'], minmax=(min, max))
    print(f'Results: {test_mae}, {test_mse}, {test_mae_norm}, {test_mse_norm}')

    # y_pred, y = inference_new(model=model, test_dl=test_dataloader, device=run_config['device'], minmax=(min, max))
    # if params.get('plot_output_folder', None) is not None:
    #     plot_predictions(predictions, test_y, test_meta, output_folder=params.get('plot_output_folder', None))


def optuna_train_trial(trial: optuna.trial.Trial, params: dict, data: dict):
    params = params.copy()

    estimators = trial.suggest_categorical('estimators', choices=[70, 80, 90, 100, 120])
    max_depth = trial.suggest_categorical('max_depth', choices=[2, 4, 8])
    learning_rate = trial.suggest_categorical('learning_rate', choices=[1e-3, 5e-2, 1e-2])
    objective = trial.suggest_categorical('objetive', choices=['reg:absoluteerror', 'reg:squarederror'])

    x_magnitudes = params['x_magnitudes']

    params.update({
        'estimators': estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'objective': objective,
        'x_magnitudes': x_magnitudes
    })


    return train_model(params=params, data=data)


def optuna_train(params):
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

    optuna_train_with_params = partial(optuna_train_trial, params=params, data=data)
    study = optuna.create_study(direction="minimize", study_name=PROJECT_NAME, pruner=optuna.pruners.MedianPruner())
    study.optimize(optuna_train_with_params, n_trials=100)


if __name__=="__main__":
    raise RuntimeError("This is a library, can't be launched as a script")
