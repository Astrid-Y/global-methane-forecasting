import numpy as np
import pytest
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch4forecast.model.buildmodel import error_metrics, BasicModel, BuildModel


def test_error_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    expected_mse = mean_squared_error(y_true, y_pred)
    expected_rmse = np.sqrt(expected_mse)
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_mape = mean_absolute_percentage_error(y_true, y_pred)
    expected_r2 = r2_score(y_true, y_pred)
    expected_result = np.array(
        [expected_mae, expected_mse, expected_rmse, expected_mape, expected_r2]
    )

    result = error_metrics(y_true, y_pred)

    np.allclose(
        result, expected_result, rtol=1e-5, atol=1e-5
    ), "Mismatched result of error metrics."


def test_BasicModel():
    input_tensor = torch.randn(2, 8, 7, 9, 18)
    expected_output = input_tensor[:, :, -1:, :, :]
    model = BasicModel()
    output = model(input_tensor)
    np.allclose(output, expected_output, rtol=1e-5, atol=1e-5)


class Test_BuildModel:

    def setup_class(self):
        # Create a model object
        args = SimpleNamespace(
            model="ConvLSTM",
            num_layers=2,
            hidden_dim=64,
            kernel_size=(3, 3),
            input_dim=7,
            output_dim=1,
            lr=0.001,
            device="cpu",
            pretrained=None,
        )
        self.buildmodel = BuildModel(args)

        # Create a simple dataset
        X = torch.randn(10, 8, 7, 9, 18)
        y = torch.randn(10, 8, 1, 9, 18)
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=2)

        # Define optimizer and criterion
        self.optimizer = Adam(self.buildmodel.model.parameters())
        self.criterion = nn.MSELoss()

    def test_train_epoch(self):
        train_loss, train_err = self.buildmodel.train_epoch(
            self.dataloader, self.optimizer, self.criterion
        )
        assert isinstance(train_loss, float), "train_loss should be a float"
        assert isinstance(train_err, np.ndarray), "train_err should be a numpy array"
        assert train_err.shape == (5,), f"the shape of train_err should be 5"

    def test_validate_epoch(self):
        val_loss, val_err = self.buildmodel.validate_epoch(
            self.dataloader, self.criterion
        )
        assert isinstance(val_loss, float), "train_loss should be a float"
        assert isinstance(val_err, np.ndarray), "train_err should be a numpy array"
        assert val_err.shape == (5,), f"the shape of train_err should be 5"
