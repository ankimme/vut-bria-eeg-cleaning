#!/usr/bin/env python3

import os
import datetime
import pickle

import torch
from torch import optim
from torch.nn import MSELoss
from torch.cuda.amp import GradScaler

from dataloader import create_dataloader
from enums import NoiseTypeEnum
from model_factory import ModelFactory


def create_model() -> torch.nn.Module:
    model = ModelFactory.LSTM_01()
    model = model.to(device)
    return model


def create_run_dir(results_dir: str) -> str:
    run_id = f"run-{datetime.datetime.now().isoformat()}"
    run_dir = f"{results_dir}/{run_id}"

    os.mkdir(f"{run_dir}")
    os.mkdir(f"{run_dir}/model")
    os.mkdir(f"{run_dir}/plots")
    os.mkdir(f"{run_dir}/loss_history")

    return run_dir


def train_single_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler,
    dl_train: torch.utils.data.DataLoader,
    loss_history: dict,
    epoch: int,
) -> dict:
    model.train()
    running_loss = 0.0
    for input, target in dl_train:
        # clear old gradients
        optimizer.zero_grad()

        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    # divide the running loss by the number of samples because the loss is summed over all samples (reduction="sum")
    running_loss /= len(dl_train.dataset)

    if epoch == 30:
        optimizer.param_groups[0]["initial_lr"] /= 10
    print(f"Epoch: {epoch}")
    print(f"Train loss: {running_loss:.2f}")
    loss_history["train"].append(running_loss)

    scheduler.step()
    return loss_history


def evaluate_single_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dl_test: torch.utils.data.DataLoader,
    epoch: int,
    loss_history: dict,
    run_dir: str,
) -> dict:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, target_regression in dl_test:
            data = data.to(device)
            target_regression = target_regression.to(device)
            output_regression = model(data)
            loss = criterion(output_regression, target_regression)

            running_loss += loss.item()

    running_loss /= len(dl_test.dataset)
    print(f"Test loss: {running_loss:.2f}\n")
    loss_history["test"].append(running_loss)

    pickle.dump(
        loss_history, open(f"{run_dir}/loss_history/loss_history_{epoch}.pickle", "wb")
    )

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{run_dir}/model/epoch_{epoch}.pth")

    return loss_history


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler,
    dl_train: torch.utils.data.DataLoader,
    dl_test: torch.utils.data.DataLoader,
    results_dir: str,
    epoch_cnt: int,
):
    run_dir = create_run_dir(results_dir)

    loss_history = {
        "train": [],
        "test": [],
    }
    for epoch in range(epoch_cnt):
        loss_history = train_single_epoch(
            model,
            criterion,
            optimizer,
            scaler,
            scheduler,
            dl_train,
            loss_history,
            epoch,
        )
        loss_history = evaluate_single_epoch(
            model, criterion, dl_test, epoch, loss_history, run_dir
        )
    torch.save(model.state_dict(), f"{run_dir}/model/FINAL.pth")


def main():
    BATCH_SIZE = 32
    EPOCH_CNT = 2500

    model = create_model()
    criterion = MSELoss(reduction="sum")
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.000008)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.5, 0.9))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.000_05, betas=(0.82, 0.95), weight_decay=0.000_01
    )
    scaler = GradScaler()
    dl_train, dl_test = create_dataloader(
        BATCH_SIZE,
        [
            NoiseTypeEnum.NONE,  # train with positive examples
            NoiseTypeEnum.EOG,
            NoiseTypeEnum.EMG,
        ],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=0.000_000_01
    )

    print("Start training")
    train(
        model,
        criterion,
        optimizer,
        scaler,
        scheduler,
        dl_train,
        dl_test,
        "./results",
        EPOCH_CNT,
    )


if __name__ == "__main__":
    # cuda setup
    cuda_available = torch.cuda.is_available()

    # # GPU Setup
    device = torch.device("cuda:0" if cuda_available else "cpu")

    if cuda_available:
        print("Using: " + str(torch.cuda.get_device_name(device)))
    else:
        print("Using: CPU")
    main()
