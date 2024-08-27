import numpy as np
import pandas as pd
import re


def read_loss(log_file_path):
    """
    Reads and parses training and validation loss values from a log file.

    Parameters
    ----------
    log_file_path : str
        The path to the log file containing training and validation loss information.

    Returns
    -------
    epochs : list of int
        A list of epoch numbers extracted from the log file.
    results : dict of list of float
        A dictionary containing two keys: "Train loss" and "Val loss". Each key maps to a list of floats
        representing the training and validation losses for each epoch.
    """
    log_pattern = re.compile(
        r"Epoch (\d+)/\d+, Train Loss: ([\d.]+),.*Val Loss: ([\d.]+).*"
    )

    epochs = []
    train_losses = []
    val_losses = []

    with open(log_file_path, "r") as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
    results = {"Train loss": train_losses, "Val loss": val_losses}
    return epochs, results


def read_loss_ml(log_file_path):
    """
    Reads and parses multiple loss metrics from a log file.

    Parameters
    ----------
    log_file_path : str
        The path to the log file containing training and validation loss information. The log file is expected
        to have lines formatted with various loss metrics such as "Train total_loss", "Val total_loss",
        "Train p_loss", "Val p_loss", "Train e_loss", and "Val e_loss".

    Returns
    -------
    epochs : list of int
        A list of epoch numbers extracted from the log file.
    loss : dict of list of float
        A dictionary where each key corresponds to a specific loss metric (e.g., "Train total_loss",
        "Val total_loss", etc.), and each value is a list of floats representing the respective loss values
        for each epoch.
    """
    log_pattern = re.compile(
        r"Epoch (\d+)/\d+, Train total_loss: ([\d.e+-]+),.* Val total_loss: ([\d.e+-]+),.* Train p_loss: ([\d.e+-]+),.* Val p_loss: ([\d.e+-]+),.* Train e_loss: ([\d.e+-]+),.* Val e_loss: ([\d.e+-]+).*,"
    )

    epochs = []
    keys = [
        "Train total_loss",
        "Val total_loss",
        "Train p_loss",
        "Val p_loss",
        "Train e_loss",
        "Val e_loss",
    ]
    loss = {}
    for k in keys:
        loss[k] = []

    with open(log_file_path, "r") as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                for i, k in enumerate(keys):
                    loss[k].append(float(match.group(2 + i)))
    return epochs, loss


def read_eval(log_file_path, stds=True):
    """
    Reads and parses evaluation metrics from a log file.

    Parameters
    ----------
    log_file_path : str
        The path to the log file containing evaluation metrics such as MSE, RMSE, R2, and optionally standard deviation (std).
    stds : bool, optional
        If True, the function expects and parses the standard deviation (std) from the log file. If False, the std values are not expected. Default is True.

    Returns
    -------
    days : list of int
        A list of day numbers extracted from the log file.
    results : dict of list of float
        A dictionary where each key corresponds to a specific evaluation metric (e.g., "mse", "rmse", "r2", "std"), and each value is a list of floats representing the respective metric values for each day.
    """
    if stds:
        log_pattern = re.compile(
            r"Day (\d+):.* mse: ([\d.e+-]+),.* rmse: ([\d.e+-]+),.* r2: ([\d.e+-]+),.* std: ([\d.e+-]+),"
        )
        keys = [
            "mse",
            "rmse",
            "r2",
            "std",
        ]
    else:
        log_pattern = re.compile(
            r"Day (\d+):.* mse: ([\d.e+-]+),.* rmse: ([\d.e+-]+),.* r2: ([\d.e+-]+),"
        )
        keys = [
            "mse",
            "rmse",
            "r2",
        ]
    days = []
    results = {}
    for k in keys:
        results[k] = []

    with open(log_file_path, "r") as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                days.append(int(match.group(1)))
                for i, k in enumerate(keys):
                    results[k].append(float(match.group(2 + i)))
    return days, results
