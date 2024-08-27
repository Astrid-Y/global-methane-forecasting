"""
Author: Boyang Hu
GitHub Username: edsml-bh223 

Description:
    Functions of training, testing, predicting models.
"""

import sys
import os
import numpy as np
import random
import dask.array as da
import xarray as xr
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import logging
import psutil
from ch4forecast.model.convLSTM import CALSTMNet, TALSTMNet


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return True


def error_metrics(y_true, y_pred):
    """
    Calculate error metrics for model evaluation.

    Parameters
    ----------
    y_true : array-like
        The ground truth target values.
    y_pred : array-like
        The predicted values by the model.

    Returns
    -------
    np.array
        An array containing the following metrics:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Mean Absolute Percentage Error (MAPE)
        - R-squared (R2) Score
    """
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    res = np.array([mae, mse, rmse, mape, r2])
    return res


class BasicModel(nn.Module):  # persistent forecasting
    """Persistent model for methane concentration forecasting."""

    def __init__(self):
        super(BasicModel, self).__init__()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, t, C, H, W].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, t, 1, H, W],
            where the last time step is repeated for all future time steps.
        """
        return x[:, :, -1:, :, :]


class BuildModel:
    """
    A class to build and configure different forecasting models based on provided arguments.

    This class initializes various forecasting models such as TALSTMNet and CALSTMNet, provides functions for training, testing, predicting.

    Parameters
    ----------
    args : Namespace
        Arguments for model configuration, including:
        - model (str): The type of model to build:
            - 'ConvLSTM': model without any attention
            - 'CALSTM': model with channel attention
            - 'SALSTM': model with spatial attention
            - 'CSLSTM': model with both spatial and channel attention (CBAM)
            - 'TALSTM': model with temporal attention
            - 'CSTALSTM': model with both CBAM and temporal attention
        - name (str): name of files to be saved.
        - input_dim (int): The number of input channels.
        - hidden_dim (list of int): The number of hidden units for each layer.
        - kernel_size (list of tuple): The kernel sizes for each layer.
        - num_layers (int): The number of layers in the model.
        - output_dim (int): The number of output channels.
        - device (str): The device to use ('cpu' or 'cuda').
        - pretrained (str): Path to a pre-trained model state dictionary (optional).
        - val_rio (float): Ratio for splitting the dataset into training and validation sets (if applicable).
        - batch_size (int): The batch size for data loading.
        - num_workers (int): Number of workers for data loading.

    Attributes
    ----------
    model : nn.Module
        The initialized model based on the specified type.
    device : torch.device or str
        The device on which the model will be placed ('cpu' or 'cuda').
    gradscaler : GradScaler
        The gradient scaler used for mixed precision training.
    evalkeys : list of str
        Keys for evaluation metrics.
    """

    def __init__(self, args):
        super().__init__()
        attnmap = {
            "ConvLSTM": None,
            "SALSTM": "SA",
            "CALSTM": "CA",
            "CSLSTM": "CS",
            "TALSTM": None,
            "CSTALSTM": "CS",
        }
        if args.model in ["TALSTM", "CSTALSTM"]:
            self.model = TALSTMNet(
                args.input_dim,
                args.hidden_dim,
                args.kernel_size,
                args.num_layers,
                args.output_dim,
                attention=attnmap[args.model],
            )
        elif args.model in ["ConvLSTM", "CALSTM", "SALSTM", "CSLSTM"]:
            self.model = CALSTMNet(
                args.input_dim,
                args.hidden_dim,
                args.kernel_size,
                args.num_layers,
                args.output_dim,
                attention=attnmap[args.model],
            )
        else:
            self.model = BasicModel()

        if args.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        if args.pretrained:
            self.model.load_state_dict(
                torch.load(args.pretrained, map_location=self.device)
            )
        self.args = args
        self.gradscaler = GradScaler()
        self.evalkeys = ["mae", "mse", "rmse", "mape", "r2"]

    def compute_resource(self, data):
        """
        Compute and print resource usage for the model.

        Parameters
        ----------
        data : tuple
            A tuple containing input and target tensors used for computing resource usage.

        Prints
        -------
        Trainable parameters count and memory usage (CUDA and RAM).

        References
        ----------
        .. [1] https://chatgpt.com/share/6059226b-612e-4d55-8c69-e94c7159c488
        """
        num_param = num_param = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Trainable parameters: ", num_param)
        with torch.no_grad():
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            self.model(x)
            mem_cuda = torch.cuda.memory_allocated(self.device) / 1024**2
            ram_usage = psutil.Process().memory_info().rss / 1024**2
        print(f"Cuda memory usage: {mem_cuda} MB")
        print(f"RAM usage: {ram_usage} MB")

    def train_epoch(self, data, optimizer, criterion):
        """
        Perform a single training epoch.

        Parameters
        ----------
        data : DataLoader
            DataLoader providing batches of input and target data for training.
        optimizer : torch.optim.Optimizer
            Optimizer used for updating model parameters.
        criterion : nn.Module
            Loss function used to compute the training loss.

        Returns
        -------
        train_loss : float
            Average training loss for the epoch.
        train_err : np.ndarray
            Array of evaluation metrics (mae, mse, rmse, mape, r2) for the training data.
        """
        self.model.train()
        train_loss = 0.0
        train_err = np.zeros(len(self.evalkeys))
        for X, y in data:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            with autocast():
                output = self.model(X)
                loss = criterion(output, y)
            self.gradscaler.scale(loss).backward()  # amplify gradient
            self.gradscaler.step(optimizer)
            self.gradscaler.update()
            train_loss += loss.item() * X.shape[0]
            batch_err = error_metrics(
                y.detach().cpu().numpy(), output.float().detach().cpu().numpy()
            )
            train_err += batch_err * X.shape[0]

            del X, y, batch_err, loss
            torch.cuda.empty_cache()

        train_loss /= len(data.dataset)
        train_err /= len(data.dataset)
        return train_loss, train_err

    def validate_epoch(self, data, criterion):
        """
        Perform a single validation epoch.

        Parameters
        ----------
        data : DataLoader
            DataLoader providing batches of input and target data for validation.
        criterion : nn.Module
            Loss function used to compute the validation loss.

        Returns
        -------
        val_loss : float
            Average validation loss for the epoch.
        val_err : np.ndarray
            Array of evaluation metrics (mae, mse, rmse, mape, r2) for the validation data.
        """
        self.model.eval()
        val_loss = 0.0
        val_err = np.zeros(len(self.evalkeys))
        for X, y in data:
            with torch.no_grad():
                X, y = X.to(self.device), y.to(self.device)
                with autocast():
                    output = self.model(X)
                    loss = criterion(output, y)
                val_loss += loss.item() * X.shape[0]
                batch_err = error_metrics(y.cpu().numpy(), output.float().cpu().numpy())
                val_err += batch_err * X.shape[0]

            del X, y, batch_err, loss
            torch.cuda.empty_cache()

        val_loss /= len(data.dataset)
        val_err /= len(data.dataset)
        return val_loss, val_err

    def train(self, trainloader, valloader):
        """
        Train the model over multiple epochs and validate.

        Save the training logs and model checkpoints. The saved path is controlled by args.output_path and args.name.

        Parameters
        ----------
        trainloader : DataLoader
            DataLoader providing batches of input and target data for training.
        valloader : DataLoader
            DataLoader providing batches of input and target data for validation.
        """
        model_path = os.path.join(self.args.output_path, "checkpoints")
        log_path = os.path.join(self.args.output_path, "train_logs")
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)

        logging.basicConfig(
            filename=os.path.join(log_path, f"{self.args.name}_{self.args.seqlen}.txt"),
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
        )

        if self.args.pretrained:
            pre_epoch = int(self.args.pretrained.split("_")[-1].split(".")[0])
        else:
            pre_epoch = 0

        print("Start training...")
        for epoch in tqdm(range(pre_epoch, self.args.n_epoch)):
            train_loss, train_err = self.train_epoch(trainloader, optimiser, criterion)
            val_loss, val_err = self.validate_epoch(valloader, criterion)

            torch.cuda.empty_cache()  # Empty cache

            log_i = f"Epoch {epoch+1}/{self.args.n_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            for i, k in enumerate(self.evalkeys):
                log_i += f"Train {k}: {train_err[i]:.4e}, Val {k}: {val_err[i]:.4e}, "
            logging.info(log_i)

            # Save model every 3 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    model_path, f"{self.args.name}_{self.args.seqlen}_{epoch+1}.pth"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model saved at {checkpoint_path}")

    def test_roll(self, data):
        """
        Evaluate the model's performance by rolling predictions over multiple days.

        Save the test results. The saved path is controlled by args.output_path and args.name.

        Parameters
        ----------
        data : nn.utils.data.dataset
            Dataset providing input and target data for testing.

        Returns
        -------
        err : np.ndarray
            Array of evaluation metrics averaged over all test samples and days.
        """
        # data: nn.utils.data.dataset
        days = self.args.test_days
        seqlen = self.args.seqlen
        self.model.eval()
        err = np.zeros((days, len(self.evalkeys)))
        mses = np.zeros((len(data) - days + 1, days))
        print(f"Evaluating on testset for next {days} days...")
        for idx in tqdm(range(len(data) - days + 1)):
            pred = torch.tensor(data[idx][0][:, -1:, :, :]).to(self.device)
            for i in range(days):
                x, y = data[idx + i]
                x = torch.tensor(x).to(self.device)
                x[:, -1:, :, :] = pred
                with torch.no_grad():
                    with autocast():
                        pred = self.model(x.unsqueeze(0))
                erri = error_metrics(y, pred.float().cpu().numpy())
                mses[idx][i] = erri[1]
                err[i] += erri
        err /= len(data) - days + 1
        mse_std = np.std(mses, axis=0).reshape(-1, 1)
        err = np.concatenate([err, mse_std], axis=1)

        output_path = os.path.join(self.args.output_path, "test_logs")
        os.makedirs(output_path, exist_ok=True)
        with open(
            os.path.join(output_path, f"{self.args.name}_{seqlen}x{days}.txt"),
            "w",
        ) as f:
            for d in range(days):
                res = f"Day {d+1}: "
                for i, k in enumerate(self.evalkeys + ["std"]):
                    res += f"{k}: {err[d][i]:.4e}, "
                print(res)
                f.write(res + "\n")
        return err

    def predict(self, data, timeindex: int | list, days: int):
        """
        Generate predictions for a specified number of days starting from given time indices.

        Save the predictions. The saved path is controlled by args.output_path and args.name.

        Parameters
        ----------
        data : nn.utils.data.dataset
            Dataset providing input and target data for prediction.
        timeindex : int or list
            Number of random time indices to sample or specific list of time indices for predictions.
        days : int
            Number of days to predict.

        Returns
        -------
        predictions : np.ndarray
            Array of predicted values for the specified number of days and time indices.
        """
        if days >= len(data):
            raise ValueError(
                f"days must be an integer and must be between 0 and the length of data({len(data)})."
            )
        if isinstance(timeindex, int):
            timeindex = sorted(random.sample(range(len(data) - days), timeindex))
        else:
            timeindex = sorted(timeindex)

        seqlen = self.args.seqlen
        self.model.eval()
        h, w = data[0][0].shape[-2:]
        predictions = np.zeros((len(timeindex) * seqlen, days, h, w))
        print(f"Predict for next {days} days...")
        for idx, timeidx in tqdm(enumerate(timeindex)):
            pred = torch.tensor(data[timeidx][0][:, -1:, :, :]).to(self.device)
            for i in range(days):
                x, y = data[timeidx + i]
                x = torch.tensor(x).to(self.device)
                x[:, -1:, :, :] = pred
                with torch.no_grad():
                    with autocast():
                        pred = self.model(x.unsqueeze(0))
                for t in range(seqlen):
                    predictions[idx * seqlen + t][i] = (
                        pred.float().cpu().numpy()[0, t, 0, :, :]
                    )
        data.writedata(predictions, timeindex, self.args.output_path, self.args.name)
        return predictions
