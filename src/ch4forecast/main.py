"""
Author: Boyang Hu
GitHub Username: edsml-bh223 

Description:
    Main file of the ch4forecast project.
"""

import argparse
import sys
import os
import torch
import warnings

from ch4forecast.model.buildmodel import BuildModel, set_seed
from ch4forecast.utils.dataloader import (
    load_train_data,
    load_test_data,
)


def parse_tuple(arg):
    """
    Parses a string representing a tuple of integers and converts it into an actual tuple of integers.

    Parameters
    ----------
    arg : str
        A string representing a tuple, typically provided as an argument in the format "x,y" or "(x,y)",
        where x and y are integers.

    Returns
    -------
    tuple of int
        A tuple containing the integers extracted from the input string.

    Raises
    ------
    argparse.ArgumentTypeError
        If the input string cannot be parsed into a tuple of integers, an `ArgumentTypeError` is raised
        with an appropriate error message.
    """
    try:
        # Remove parentheses if present
        arg = arg.strip("()")
        # Split the string by commas and convert each part to an integer
        return tuple(map(int, arg.split(",")))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {arg}")


def parse_args():
    """
    Parses command-line arguments for configuring the methane concentrations forecasting model.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments as attributes.

    Parameters
    ----------
    --name : str, optional
        To control the name of output files to be saved.
    --model : str, optional
        The model architecture to use. Choices are
            - `ConvLSTM`: model without any attention;
            - `CALSTM`: model with channel attention;
            - `SALSTM`: model with spatial attention;
            - `CSLSTM`: model with both spatial and channel attention (CBAM);
            - `TALSTM`: model with temporal attention;
            - `CSTALSTM`: model with both CBAM and temporal attention.
        Default is "ConvLSTM".
    --data_path : str, optional
        The path to the directory containing the dataset. Default is "../data/resample".
    --output_path : str, optional
        The path to the directory where output files will be saved. Default is "../output".
    --seqlen : int, optional
        The sequence length, representing the number of time steps used to build MethaneDataset. Default is 8.
    --val_rio : float, optional
        The ratio of the dataset to be used for validation. Default is 0.2.
    --batch_size : int, optional
        The batch size used during training. Default is 16.
    --num_workers : int, optional
        The number of worker threads for loading data. Default is 4.
    --n_epoch : int, optional
        The number of training epochs. Default is 30.
    --num_layers : int, optional
        The number of hidden layers in the LSTM architecture. Default is 3.
    --hidden_dim : int, optional
        The number of hidden dimensions in each layer of the model. Default is 64.
    --kernel_size : tuple of int, optional
        The size of the convolutional kernels. Default is (7, 7). Use a comma-separated string
        to specify, e.g., "3,3".
    --input_dim : int, optional
        The number of input channels/features. Default is 7.
    --output_dim : int, optional
        The number of output channels/features. Default is 1.
    --lr : float, optional
        The learning rate for the optimizer. Default is 1e-3.
    --device : str, optional
        The device to run the model on. Choices are "cuda" or "cpu". Default is "cuda".
    --pretrained : str, optional
        The path to a pretrained model checkpoint. Default is None.
    --train : bool, optional
        If specified, the model will be trained. Default is False.
    --test : bool, optional
        If specified, the model will be tested. Default is False.
    --predict : bool, optional
        If specified, the model will make predictions. Default is False.
    --test_days : int, optional
        The number of days of rolling forecasting to test. Parameter is valid only when `test` is true. Default is 14.
    --predict_samples : int, optional
        The number of prediction samples to generate. Parameter is valid only when `predict` is true. Default is 20.
    --predict_days : int, optional
        The number of days ahead to predict. Parameter is valid only when `predict` is true. Default is 14.
    --calculation : bool, optional
        If specified, compute and print resource usage for the model. Default is False.
    """
    parse = argparse.ArgumentParser(
        description="Methane concentrations forecasting model"
    )
    # add parameters
    parse.add_argument("--name", type=str, default="test")
    parse.add_argument(
        "--model",
        default="ConvLSTM",
        choices=[
            "ConvLSTM",
            "SALSTM",
            "CALSTM",
            "CSLSTM",
            "TALSTM",
            "CSTALSTM",
            "basic",
        ],
    )
    parse.add_argument("--data_path", type=str, default="../data/resample")
    parse.add_argument("--output_path", type=str, default="../output")
    parse.add_argument("--seqlen", type=int, default=8)
    parse.add_argument("--val_rio", type=float, default=0.2)
    parse.add_argument("--batch_size", type=int, default=16)
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--n_epoch", type=int, default=30)
    parse.add_argument("--num_layers", type=int, default=3)
    parse.add_argument("--hidden_dim", type=int, default=64)
    parse.add_argument("--kernel_size", type=parse_tuple, default=(7, 7))
    parse.add_argument("--input_dim", type=int, default=7)
    parse.add_argument("--output_dim", type=int, default=1)
    parse.add_argument("--lr", type=float, default=1e-3)
    parse.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parse.add_argument("--pretrained", type=str, default=None)
    parse.add_argument("--train", action="store_true")
    parse.add_argument("--test", action="store_true")
    parse.add_argument("--predict", action="store_true")
    parse.add_argument("--test_days", type=int, default=14)
    parse.add_argument("--predict_samples", type=int, default=20)
    parse.add_argument("--predict_days", type=int, default=14)
    parse.add_argument("--calculation", action="store_true")

    args = parse.parse_args()
    return args


def main():
    """
    The main interface function of the project.
    """
    args = parse_args()
    print("n_cpu: ", os.cpu_count())
    print("CUDA available: ", torch.cuda.is_available())
    print("n_gpu: ", torch.cuda.device_count())
    if args.device == "cpu":
        warnings.filterwarnings(
            "ignore",
            message="User provided device_type of 'cuda', but CUDA is not available.",
        )
    # set_seed(42)
    model = BuildModel(args)
    print("Parameters:")
    for k, v in args.__dict__.items():
        print(f"{k} : {v}")

    if args.calculation:
        testloader = load_test_data(args)
        data = next(iter(testloader))
        model.compute_resource(data)
    if args.train:
        trainloader, valloader = load_train_data(args)
        model.train(trainloader, valloader)
    if args.test:
        testloader = load_test_data(args)
        err = model.test_roll(testloader.dataset)
    if args.predict:
        testloader = load_test_data(args)
        predictions = model.predict(
            testloader.dataset, args.predict_samples, args.predict_days
        )


if __name__ == "__main__":
    main()
