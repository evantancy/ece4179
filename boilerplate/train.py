import pprint
import time
from pathlib import Path
from test import test_model

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader

from dataset import CustomDataset
from utils import order_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
n_workers = 0
torch.backends.cudnn.benchmark = True


n_workers = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(config):
    logger = {
        "train_loss": np.zeros(config["num_epochs"]),
        "val_loss": np.zeros(config["num_epochs"]),
        "train_acc": np.zeros(config["num_epochs"]),
        "val_acc": np.zeros(config["num_epochs"]),
        "test_acc": np.zeros(config["num_epochs"]),
    }

    #### LOAD DATA ####
    b_size = config["batch_size"]

    # set to None if not specified
    train_transform = config.get("train_transform")
    val_transform = config.get("val_transform")
    test_transform = config.get("test_transform")

    train_data = CustomDataset("train", T.ToTensor())
    train_dataloader = DataLoader(
        train_data,
        batch_size=b_size,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=False,
    )

    val_data = CustomDataset("val", T.ToTensor())
    val_dataloader = DataLoader(
        val_data,
        batch_size=b_size,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=False,
    )

    test_data = CustomDataset("test", T.ToTensor())
    test_dataloader = DataLoader(
        test_data,
        batch_size=b_size,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=False,
    )

    #### INSTANTIATE MODEL ####
    net = config["model"].to(device)
    loss_function = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    if "Adam" in config["optimizer"]:
        optimizer = optim.Adam(
            net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    elif "SGD" in config["optimizer"]:
        optimizer = optim.SGD(
            net.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

    # TODO: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    # https://spell.ml/blog/lr-schedulers-and-adaptive-optimizers-YHmwMhAAACYADm6F
    if config.get("lr_scheduler") is not None:
        div_factor = config["lr_scheduler"].get("div_factor")
        div_factor = 1e4 if div_factor is None else div_factor

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            steps_per_epoch=len(train_dataloader),
            epochs=config["num_epochs"],
            final_div_factor=div_factor,
        )

    #### BEGIN TRAINING ####
    start_time = time.time()
    best_val_acc = 0

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(f"Train Transforms: {train_transform}")
    pp.pprint(f"Test Transforms: {test_transform}")
    pp.pprint(f"Val Transforms: {val_transform}")

    for j in range(config["num_epochs"]):
        ## START OF EPOCH ##
        train_loss, train_steps = 0.0, 0
        net.train()

        for batch_id, (data, label) in enumerate(train_dataloader):
            data, label = data.to(device), label.long().to(device)

            if train_dataloader.dataset.transform is None:
                data = order_tensor(data)
            if train_transform is not None:
                data = train_transform(data.cuda())

            # forwardfacecolor=fig.get_facecolor()
            with torch.cuda.amp.autocast():
                output = net(data)
                loss = loss_function(output, label)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_steps += 1

            if config.get("lr_scheduler") is not None:
                # OneCycleLR steps inside dataloader loop
                scheduler.step()

        ## END OF EPOCH ##

        # average training loss for 1 epoch
        train_loss /= train_steps

        # test model on validation dataset
        _, train_acc = test_model(net, train_dataloader, loss_function, train_transform)
        val_loss, val_acc = test_model(
            net, val_dataloader, loss_function, val_transform
        )
        _, test_acc = test_model(net, test_dataloader, loss_function, test_transform)

        logger["train_loss"][j] = train_loss
        logger["val_loss"][j] = val_loss
        logger["train_acc"][j] = train_acc
        logger["val_acc"][j] = val_acc
        logger["test_acc"][j] = test_acc

        if config["log_training"] and (j + 1) % config["log_interval"] == 0:
            print(
                f"Epoch:{j+1}/{config['num_epochs']}",
                f"Train Loss: {logger['train_loss'][j]:.4f}",
                f"Train Acc: {logger['train_acc'][j]:.4f}",
                f"Val Loss: {logger['val_loss'][j]:.4f}",
                f"Val Acc: {logger['val_acc'][j]:.4f}",
                f"Test Acc: {logger['test_acc'][j]:.4f}",
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if config["save_model"]:
                # make sure folder is created to place saved checkpoints
                path = Path.cwd() / "models" / net._name
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=False)

                # pad with appropriate number of zeros i.e. epoch 10 named as 010
                checkpoint_num = str(j + 1).zfill(len(str(config["num_epochs"])))
                model_path = f"./models/{net._name}/{net._name}_{checkpoint_num}.pt"
                torch.save(net.state_dict(), model_path)

    print(f"{config['num_epochs']} epochs took {time.time() - start_time:.2f}s")

    if config["log_training"]:
        return logger
