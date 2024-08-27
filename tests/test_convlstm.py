import numpy as np
import torch
import pytest
import torch.nn as nn

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch4forecast.model.attention import (
    SpatialAttention,
    CBAM,
    TemporalAttention,
)
from ch4forecast.model.convLSTM import ConvLSTMCell, ConvLSTM, CALSTMNet, TALSTMNet


class Test_ConvLSTMCell:

    def setup_class(self):
        self.input_dim = 3
        self.hidden_dim = 4
        self.kernel_size = (3, 3)
        self.batch_size = 2
        self.field_size = (9, 18)

        self.cell = ConvLSTMCell(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
            attention=None,
        )

        self.input_tensor = torch.randn(
            self.batch_size, self.input_dim, *self.field_size
        )

    def test_init(self):
        assert isinstance(self.cell.conv, nn.Conv2d)
        assert self.cell.ci.shape == (1, self.hidden_dim, 1, 1)
        assert self.cell.cf.shape == (1, self.hidden_dim, 1, 1)
        assert self.cell.co.shape == (1, self.hidden_dim, 1, 1)

    def test_init_hidden(self):
        h, c = self.cell.init_hidden(self.batch_size, self.field_size)
        assert h.shape == (
            self.batch_size,
            self.hidden_dim,
            *self.field_size,
        ), "Initialization of hidden state shape mismatch."
        assert c.shape == (
            self.batch_size,
            self.hidden_dim,
            *self.field_size,
        ), "Initialization of cell state shape mismatch."

    def test_forward(self):
        h_cur, c_cur = self.cell.init_hidden(self.batch_size, self.field_size)
        h_next, c_next = self.cell.forward(self.input_tensor, (h_cur, c_cur))
        assert h_next.shape == (
            self.batch_size,
            self.hidden_dim,
            *self.field_size,
        ), "Hidden state shape mismatch."
        assert c_next.shape == (
            self.batch_size,
            self.hidden_dim,
            *self.field_size,
        ), "Cell state shape mismatch."

    def test_with_CS(self):
        self.cell = ConvLSTMCell(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
            attention="CS",
        )
        assert isinstance(self.cell.attention, CBAM)
        self.test_init_hidden()
        self.test_forward()

    def test_with_CA(self):
        self.cell = ConvLSTMCell(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
            attention="CA",
        )
        assert isinstance(self.cell.attention, CBAM)
        self.test_init_hidden()
        self.test_forward()

    def test_with_SA(self):
        self.cell = ConvLSTMCell(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
            attention="SA",
        )
        assert isinstance(self.cell.attention, SpatialAttention)
        self.test_init_hidden()
        self.test_forward()


class Test_ConvLSTM:

    def setup_class(self):
        self.input_dim = 7
        self.hidden_dim = [4, 4]
        self.kernel_size = (3, 3)
        self.num_layers = 2
        self.batch_size = 2
        self.time_steps = 8
        self.height, self.width = 9, 18

        self.model = ConvLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            attention=None,
        )

    def test_init(self):
        assert len(self.model.cell_list) == self.num_layers
        assert isinstance(self.model.cell_list[0], nn.Module)
        assert isinstance(self.model.cell_list[1], nn.Module)

    def test_forward(self):
        input_tensor = torch.randn(
            self.batch_size, self.time_steps, self.input_dim, self.height, self.width
        )
        layer_output_list, last_state_list = self.model(input_tensor)

        # Check the shape of output
        assert layer_output_list.shape == (
            self.batch_size,
            self.time_steps,
            self.hidden_dim[-1],
            self.height,
            self.width,
        ), "Output shape mismatch."

        # Check the shape of hidden state and cell state
        assert last_state_list[0].shape == (
            self.batch_size,
            self.hidden_dim[-1],
            self.height,
            self.width,
        ), "Hidden state shape mismatch."
        assert last_state_list[1].shape == (
            self.batch_size,
            self.hidden_dim[-1],
            self.height,
            self.width,
        ), "Cell state shape mismatch."

    def test_hidden_state(self):
        input_tensor = torch.randn(
            self.batch_size, self.time_steps, self.input_dim, self.height, self.width
        )
        init_hidden = self.model._init_hidden(
            self.batch_size, (self.height, self.width)
        )
        _, last_state_list = self.model(input_tensor, hidden_state=init_hidden)

        assert last_state_list[0].shape == (
            self.batch_size,
            self.hidden_dim[-1],
            self.height,
            self.width,
        ), "Initialization of hidden state shape mismatch."
        assert last_state_list[1].shape == (
            self.batch_size,
            self.hidden_dim[-1],
            self.height,
            self.width,
        ), "Initialization of cell state shape mismatch."


class Test_CALSTMNet:

    def setup_class(self):
        self.input_dim = 7
        self.hidden_dim = 64
        self.kernel_size = (3, 3)
        self.num_layers = 2
        self.output_dim = 1
        self.batch_size = 2
        self.time_steps = 8
        self.height, self.width = 9, 18

        self.model = CALSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            attention=None,
        )

    def test_init(self):
        assert isinstance(self.model.lstm, ConvLSTM)
        assert isinstance(self.model.conv, nn.Conv2d)
        assert self.model.conv.out_channels == self.output_dim

    def test_forward(self):
        input_tensor = torch.randn(
            self.batch_size, self.time_steps, self.input_dim, self.height, self.width
        )
        output_tensor = self.model(input_tensor)
        assert output_tensor.shape == (
            self.batch_size,
            self.time_steps,
            self.output_dim,
            self.height,
            self.width,
        ), "Output shape mismatch."


class Test_TALSTMNet:

    def setup_class(self):
        self.input_dim = 7
        self.hidden_dim = 64
        self.kernel_size = (3, 3)
        self.num_layers = 2
        self.output_dim = 1
        self.batch_size = 2
        self.time_steps = 8
        self.height, self.width = 9, 18

        self.model = TALSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            attention=None,
        )

    def test_init(self):
        assert isinstance(self.model.lstm, ConvLSTM)
        assert isinstance(self.model.att_layer, TemporalAttention)

    def test_forward(self):
        input_tensor = torch.randn(
            self.batch_size, self.time_steps, self.input_dim, self.height, self.width
        )
        output_tensor = self.model(input_tensor)
        assert output_tensor.shape == (
            self.batch_size,
            self.time_steps,
            self.output_dim,
            self.height,
            self.width,
        ), "Output shape mismatch."
