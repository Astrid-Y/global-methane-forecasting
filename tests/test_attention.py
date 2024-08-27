import numpy as np
import torch
import pytest
import torch.nn as nn

from ch4forecast.model.attention import (
    SpatialAttention,
    ChannelAttention,
    CBAM,
    TemporalAttention,
)


class Test_SpatialAttention:

    def setup_method(self):
        self.input_tensor = torch.randn(2, 3, 9, 18)
        self.sa = SpatialAttention(kernel_size=7)

    def test_initialization(self):
        assert isinstance(self.sa.conv1, nn.Conv2d)
        assert isinstance(self.sa.activate, nn.Sigmoid)

    def test_forward(self):
        output = self.sa(self.input_tensor)
        # Check if the output shape is [B, 1, H, W]
        assert output.shape == (
            self.input_tensor.shape[0],
            1,
            self.input_tensor.shape[2],
            self.input_tensor.shape[3],
        )
        # Check if the output value between 0 and 1 (after sigmoid)
        assert torch.all(output >= 0) and torch.all(
            output <= 1
        ), "Output must be between 0 and 1."

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_forward_with_different_kernel_sizes(self, kernel_size):
        sa = SpatialAttention(kernel_size=kernel_size)
        output = sa(self.input_tensor)
        assert output.shape == (
            self.input_tensor.shape[0],
            1,
            self.input_tensor.shape[2],
            self.input_tensor.shape[3],
        ), "Output shape mismatch."


class Test_ChannelAttention:

    def setup_method(self):
        self.channels = 64
        self.input_tensor = torch.randn(2, self.channels, 9, 18)
        self.ca = ChannelAttention(channels=self.channels)

    def test_initialization(self):
        assert isinstance(self.ca.mlp, nn.Sequential)
        assert len(self.ca.mlp) == 4
        assert isinstance(self.ca.mlp[0], nn.Flatten)
        assert isinstance(self.ca.mlp[1], nn.Linear)
        assert isinstance(self.ca.mlp[2], nn.ReLU)
        assert isinstance(self.ca.mlp[3], nn.Linear)

    def test_forward(self):
        output = self.ca(self.input_tensor)
        # Check if the output shape is the same as the input
        assert output.shape == self.input_tensor.shape, "Output shape mismatch."
        # Check if the output value between 0 and 1 (after sigmoid)
        assert torch.all(output >= 0) and torch.all(
            output <= 1
        ), "Output must be between 0 and 1."

    @pytest.mark.parametrize("reduction_ratio", [4, 8, 16])
    def test_forward_with_different_reduction_ratios(self, reduction_ratio):
        ca = ChannelAttention(channels=self.channels, reduction_ratio=reduction_ratio)
        output = ca(self.input_tensor)
        assert output.shape == self.input_tensor.shape, "Output shape mismatch."


class Test_CBAM:

    def setup_method(self):
        self.channels = 64
        self.input_tensor = torch.randn(2, self.channels, 9, 18)
        self.cbam = CBAM(channels=self.channels)

    def test_initialization(self):
        assert isinstance(self.cbam.channel_attn, ChannelAttention)
        if self.cbam.spatial_attn is not None:
            assert isinstance(self.cbam.spatial_attn, SpatialAttention)
        else:
            assert self.cbam.spatial_attn is None

    def test_forward_with_spatial(self):
        output = self.cbam(self.input_tensor)
        assert output.shape == (
            self.input_tensor.shape[0],
            1,
            self.input_tensor.shape[2],
            self.input_tensor.shape[3],
        ), "Output shape mismatch."

    def test_forward_without_spatial(self):
        cbam_no_spatial = CBAM(channels=self.channels, spatial=False)
        output = cbam_no_spatial(self.input_tensor)
        assert output.shape == self.input_tensor.shape, "Output shape mismatch."


class Test_TemporalAttention:

    def setup_method(self):
        self.input_dim = 64
        self.hidden_dim = 32
        self.out_dim = 1
        self.batch_size = 2
        self.time_steps = 8
        self.height = 9
        self.width = 18

        self.input_tensor = torch.randn(
            self.batch_size, self.time_steps, self.input_dim, self.height, self.width
        )
        self.temporal_attention = TemporalAttention(
            self.input_dim, self.hidden_dim, self.out_dim
        )

    def test_initialization(self):
        assert isinstance(self.temporal_attention.query, nn.Conv2d)
        assert isinstance(self.temporal_attention.key, nn.Conv2d)
        assert isinstance(self.temporal_attention.value, nn.Conv2d)
        assert isinstance(self.temporal_attention.softmax, nn.Softmax)

        # Check the number of input and output channels at each layer
        assert self.temporal_attention.query.in_channels == self.input_dim
        assert self.temporal_attention.query.out_channels == self.hidden_dim
        assert self.temporal_attention.key.in_channels == self.input_dim
        assert self.temporal_attention.key.out_channels == self.hidden_dim
        assert self.temporal_attention.value.in_channels == self.input_dim
        assert self.temporal_attention.value.out_channels == self.out_dim

    def test_forward(self):
        output = self.temporal_attention(self.input_tensor)
        expected_shape = (
            self.batch_size,
            self.time_steps,
            self.out_dim,
            self.height,
            self.width,
        )
        assert output.shape == expected_shape, "Output shape mismatch."

    @pytest.mark.parametrize("hidden_dim", [16, 32, 64])
    def test_forward_with_different_hidden_dims(self, hidden_dim):
        temporal_attention = TemporalAttention(self.input_dim, hidden_dim, self.out_dim)
        output = temporal_attention(self.input_tensor)
        expected_shape = (
            self.batch_size,
            self.time_steps,
            self.out_dim,
            self.height,
            self.width,
        )
        assert output.shape == expected_shape, "Output shape mismatch."
