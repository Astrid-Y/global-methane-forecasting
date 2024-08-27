"""
Author: Boyang Hu
GitHub Username: edsml-bh223 

Description:
    Build the dataset to load methane data.
"""

import sys
import os
import numpy as np
import dask.array as da
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler


class MethaneDataset(Dataset):
    """
    Dataset for methane data, including emissions and meteorological variables.

    Parameters
    ----------
    dirpath : str
        Directory path where the dataset files are stored.
    seqlen : int, optional
        Length of the sequence to be used for training or testing. Default is 8.
    choice : str, optional
        Choice of dataset split, either "train" or "test". If None, uses all data. Default is None.

    Attributes
    ----------
    data : xarray.Dataset
        Dataset containing meteorological data and methane column data.
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used for normalizing the target variable.
    seqlen : int
        Length of the sequence in one time window .
    interval : int
        Interval between time windows.
    """

    def __init__(self, dirpath, seqlen=8, choice=None):
        self.data = self._loaddata(dirpath)
        if choice == "test":
            start_date = "2018-01-01"
            end_data = "2018-12-31"
            self.data = self.data.sel(time=slice(start_date, end_data))
        elif choice == "train":
            start_date = "2015-01-01"
            end_data = "2017-12-31"
            self.data = self.data.sel(time=slice(start_date, end_data))
        self.interval = seqlen
        self.seqlen = seqlen
        self.data, self.scaler = self._min_max_scale(self.data)
        self.scaler = self.scaler["cc"]  # store the scaler for target

    def _loaddata(self, dirpath):
        """
        Load data from the given directory path.

        Parameters
        ----------
        dirpath : str
            Directory path where the dataset files are stored.

        Returns
        -------
        xarray.Dataset
            Dataset containing meteorological data, emission data and methane concentration data.
        """

        target = xr.open_dataset(
            os.path.join(dirpath, "total_column_methane.nc"), chunks={"time": 1}
        )
        emission = xr.open_dataset(
            os.path.join(dirpath, "emissions.nc"), chunks={"time": 1}
        )
        meteorology = xr.open_dataset(
            os.path.join(dirpath, "meteorology.nc"), chunks={"time": 1}
        )
        emission = emission["sum"]
        target = target["tc_ch4"]
        for cor in ["latitude", "longitude", "time"]:
            target[cor] = meteorology[cor]
            emission[cor] = meteorology[cor]
        meteorology["emission"] = emission
        meteorology["cc"] = target
        return meteorology

    def _min_max_scale(self, data):
        """
        Scales the data in an xarray.Dataset to the range [0, 1] using MinMaxScaler.

        Parameters
        ----------
        data : xarray.Dataset
            The input dataset containing variables to be normalized.

        Returns
        -------
        scaled_data : xarray.Dataset
            The dataset with all variables scaled to the range [0, 1].
        scalers : dict
            A dictionary where keys are variable names and values are the corresponding
            MinMaxScaler objects used to scale each variable.
        """

        scalers = {}
        for var in data.data_vars:
            var_value = data[var].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            var_scale = scaler.fit_transform(var_value).reshape(data[var].shape)
            data[var].data = var_scale
            scalers[var] = scaler
        return data, scalers

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples.
        """
        return (len(self.data["time"]) - 2 * self.seqlen + 1) // self.interval

    def __getitem__(self, idx):
        """
        Get a sample and its target from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            - sample: np.ndarray
                Feature data with shape [t, C, H, W].
            - target: np.ndarray
                Target data with shape [t, 1, H, W].
        """
        sample, target = [], []
        for i in range(self.seqlen):
            target.append(
                [self.data["cc"][idx * self.interval + i + self.seqlen].values]
            )
            var_list = []
            for var in self.data.data_vars:
                var_list.append(self.data[var][idx * self.interval + i].values)
            sample.append(var_list)
        sample = np.array(sample, dtype=np.float32)  # [t,C,H,W]
        target = np.array(target, dtype=np.float32)  # [t,1,H,W]
        return sample, target

    def writedata(self, prediction, timeindex, outpath, name):
        """
        Write the methane prediction to a NetCDF file.

        Parameters
        ----------
        prediction : np.ndarray
            Array of shape (timepoints, rolling_days, H, W) containing predicted data.
        timeindex : list or np.ndarray
            List or array of time indices corresponding to the predictions.
        outpath : str
            Path to the directory where the NetCDF file will be saved.
        name : str
            Name for the output file.

        Raises
        ------
        ValueError
            If the shape of the `prediction` array does not have 4 dimensions.
        """
        if len(prediction.shape) != 4:
            raise ValueError(
                "Input array should have 4 dimensions(timepoints, rolling_days, H, W)."
            )
        pred_shape = prediction.shape
        prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))
        prediction = prediction.reshape(pred_shape)
        # select timeindex
        times = []
        for i in range(self.seqlen):
            times.append(np.array(timeindex) * self.seqlen + i)
        times = sorted(np.concatenate(times, axis=0))
        times = self.data["time"][times]
        latitudes = self.data["latitude"]
        longitudes = self.data["longitude"]
        prediction_nc = {}
        for d in range(pred_shape[1]):
            prediction_nc[f"d{d+1}"] = xr.DataArray(
                prediction[:, d, :, :],
                coords=[times, latitudes, longitudes],
                dims=["time", "latitude", "longitude"],
                name=f"d{d+1}",
                attrs={
                    "units": "kg m-2",
                    "long_name": f"Day{d+1} predicted methane concentrations",
                },
            )
        prediction_nc = xr.Dataset(prediction_nc)
        outpath = os.path.join(outpath, "predictions")
        os.makedirs(outpath, exist_ok=True)
        outpath = os.path.join(outpath, f"{name}_{self.seqlen}x{pred_shape[1]}.nc")
        prediction_nc.to_netcdf(outpath)
        print(f"Predictions saved to {outpath}")


def split_dataset(dataset, split_rio=0.2):
    """
    Split a dataset into training and testing subsets.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to be split.
    split_rio : float, optional
        Ratio of the dataset to be used for testing (default is 0.2). The remaining data is used for training.

    Returns
    -------
    trainset : torch.utils.data.Subset
        The subset of the dataset used for training.
    testset : torch.utils.data.Subset
        The subset of the dataset used for testing.
    """
    split_idx = int(len(dataset) * (1 - split_rio))
    indices = list(range(len(dataset)))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    trainset = Subset(dataset, train_idx)
    testset = Subset(dataset, test_idx)
    return trainset, testset


def load_train_data(args):
    """
    Load training and validation data.

    Parameters
    ----------
    args : Namespace
        Arguments containing data paths, sequence length, validation ratio, batch size, and number of workers.

    Returns
    -------
    trainloader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    valloader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    """
    print("Start loading training data...")
    trainset = MethaneDataset(args.data_path, args.seqlen, choice="train")
    trainset, valset = split_dataset(trainset, args.val_rio)
    print("Length of train set is: ", len(trainset))
    print("Length of validate set is: ", len(valset))
    # x=[B,t,C,H,W], y=[B,t,H,W]
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    return trainloader, valloader


def load_test_data(args):
    """
    Load testing data.

    Parameters
    ----------
    args : Namespace
        Arguments containing data paths, sequence length, batch size, and number of workers.

    Returns
    -------
    testloader : torch.utils.data.DataLoader
        DataLoader for the testing dataset.
    """
    print("Start loading testing data...")
    testset = MethaneDataset(args.data_path, args.seqlen, choice="test")
    args.scaler = testset.scaler
    print("Length of test set is: ", len(testset))
    # x=[B,t,C,H,W], y=[B,t,H,W]
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return testloader
