from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from tqdm.auto import tqdm

def train_one_epoch(model, train_dl, optimizer, loss_fn, epoch_number, device, wandb_log) -> tuple:

    model.train(True)

    running_loss = 0.
    running_mae = 0.
    running_mse = 0.

    MAE: torchmetrics = torchmetrics.MeanAbsoluteError().to(device)
    MSE: torchmetrics = torchmetrics.MeanSquaredError().to(device)

    n_steps: int = int(len(train_dl))

    with tqdm(total=n_steps, leave=True) as pbar2:
        for i, (x, y) in enumerate(train_dl):
            x_batch = x.float().to(device)
            y_batch = y.float().to(device)

            optimizer.zero_grad()

            outputs = model(x_batch)

            loss = loss_fn(outputs, y_batch.view(outputs.size()))

            mae = MAE(outputs, y_batch.view(outputs.size()))
            mse = MSE(outputs, y_batch.view(outputs.size()))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mae += mae.item()
            running_mse += mse.item()

            wandb_log({"loss_train": loss.item(), "mae_train": mae.item(), "mse_train": mse.item(), "global_step": i+n_steps*epoch_number},
                      step=i+n_steps*epoch_number)
            pbar2.set_postfix(loss_train=loss.item(), mae_train=mae.item())
            pbar2.update(1)

    return running_loss / (i + 1), running_mae / (i + 1), running_mse / (i + 1)


def valid_one_epoch(model, valid_dl, loss_fn, epoch_number, device, wandb_log) -> tuple:

    model.eval()

    running_loss = 0.
    running_mae = 0.
    running_mse = 0.

    MAE: torchmetrics = torchmetrics.MeanAbsoluteError().to(device)
    MSE: torchmetrics = torchmetrics.MeanSquaredError().to(device)

    n_steps: int = int(len(valid_dl))

    with (torch.no_grad()):
        with tqdm(total=n_steps, leave=True) as pbar2:
            for i, (x, y) in enumerate(valid_dl):
                x_batch = x.float().to(device)
                y_batch = y.float().to(device)

                outputs = model(x_batch)

                loss = loss_fn(outputs, y_batch.view(outputs.size()))

                mae = MAE(outputs, y_batch.view(outputs.size()))
                mse = MSE(outputs, y_batch.view(outputs.size()))

                running_loss += loss.item()
                running_mae += mae.item()
                running_mse += mse.item()

                # wandb_log(data={"loss_valid": loss.item(), "mae_valid": mae.item(), "mse_valid": mse.item()},
                #           step=i+n_steps*epoch_number)
                pbar2.set_postfix(loss_valid=loss.item(), mae_valid=mae.item())
                pbar2.update(1)

    return running_loss / (i + 1), running_mae / (i + 1), running_mse / (i + 1)


def train(epochs: int, model, train_dl, valid_dl, loss_fn, device, optimizer, wandb_log):

    init_time: str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    with tqdm(total=epochs) as bar1:
        for e in range(epochs):
            train_loss, train_mae, train_mse = train_one_epoch(model=model,
                                                               train_dl=train_dl,
                                                               optimizer=optimizer,
                                                               loss_fn=loss_fn,
                                                               epoch_number=e,
                                                               device=device,
                                                               wandb_log=wandb_log)

            valid_loss, valid_mae, valid_mse = valid_one_epoch(model=model,
                                                               valid_dl=valid_dl,
                                                               loss_fn=loss_fn,
                                                               device=device,
                                                               epoch_number=e,
                                                               wandb_log=wandb_log)

            bar1.set_postfix(train_loss=train_loss, valid_loss=valid_loss)
            bar1.update(1)
            wandb_log({"Epochs": e, "loss_train_per_epoch": train_loss, "loss_valid_per_epoch": valid_loss})

    end_time: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    return train_loss, valid_loss, init_time, end_time


def inference(model, test_dl, device, minmax: tuple = None):

    model.eval()

    MAE: torchmetrics = torchmetrics.MeanAbsoluteError().to(device)
    MSE: torchmetrics = torchmetrics.MeanSquaredError().to(device)

    mae_metric = 0.
    mse_metric = 0.
    mae_norm_metric = 0.
    mse_norm_metric = 0.

    with (torch.no_grad()):
        with tqdm(total=len(test_dl)) as bar1:
            for i, (x, y) in enumerate(test_dl):
                x_batch = x.float().to(device)
                y_batch = y.float().to(device)
                outputs = model(x_batch)

                mae_norm = MAE(outputs, y_batch.view(outputs.size()))
                mse_norm = MSE(outputs, y_batch.view(outputs.size()))

                mae_norm_metric += mae_norm
                mse_norm_metric += mse_norm


                if minmax is not None:
                    min, max = minmax
                    y_batch = (max - min) * y_batch + min
                    outputs = (max - min) * outputs + min

                mae = MAE(outputs, y_batch.view(outputs.size()))
                mse = MSE(outputs, y_batch.view(outputs.size()))

                mae_metric += mae
                mse_metric += mse


                bar1.update(1)

    return mae_metric / (i + 1), mse_metric / (i + 1), mae_norm_metric / (i + 1), mse_norm_metric / (i + 1),


def inference_new(model, test_dl, device, minmax: tuple = None):
    model.eval()

    y_pred_list: list = []
    y_list: list = []

    with (torch.no_grad()):
        with tqdm(total=len(test_dl)) as bar1:
            for i, (x, y) in enumerate(test_dl):
                x_batch = x.float().to(device)
                y_batch = y.float().to(device)
                outputs = model(x_batch)

                if minmax is not None:
                    min, max = minmax
                    y_batch = (max - min) * y_batch + min
                    outputs = (max - min) * outputs + min

                y_list.append(y_batch)
                y_pred_list.append(outputs)

                bar1.update(1)

    return torch.cat(y_pred_list, 0), torch.cat(y_list, 0)


def get_opt(model: object, opt: str, lr: float, **kwargs: object) -> torch.optim:
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, **kwargs)
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif opt == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, **kwargs)
    elif opt == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr, **kwargs)
    elif opt == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, **kwargs)
    elif opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif opt == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=lr, **kwargs)
    elif opt == "asgd":
        return torch.optim.ASGD(model.parameters(), lr=lr, **kwargs)
    elif opt == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), lr=lr, **kwargs)
    elif opt == "rprop":
        return torch.optim.Rprop(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Optimizer {opt} not supported.")


def get_loss(loss: str, **kwargs: object) -> torch.nn.Module:
    if loss == "cross_entropy":
        return torch.nn.CrossEntropyLoss(**kwargs)
    elif loss == "mse":
        return torch.nn.MSELoss(**kwargs)
    elif loss == "l1":
        return torch.nn.L1Loss(**kwargs)
    elif loss == "poisson":
        return torch.nn.PoissonNLLLoss(**kwargs)
    elif loss == "bce":
        return torch.nn.BCELoss(**kwargs)
    elif loss == "bce_with_logits":
        return torch.nn.BCEWithLogitsLoss(**kwargs)
    elif loss == "margin_ranking":
        return torch.nn.MarginRankingLoss(**kwargs)
    elif loss == "hinge_embedding":
        return torch.nn.HingeEmbeddingLoss(**kwargs)
    elif loss == "multi_margin":
        return torch.nn.MultiMarginLoss(**kwargs)
    elif loss == "smooth_l1":
        return torch.nn.SmoothL1Loss(**kwargs)
    else:
        raise ValueError(f"Loss function {loss} not supported.")


def get_dataloaders(datasets: tuple, batch_size: int) -> tuple:
    train_dl = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_dl = DataLoader(datasets[2], batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    return train_dl, valid_dl, test_dl


