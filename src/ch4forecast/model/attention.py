"""
Author: Boyang Hu
GitHub Username: edsml-bh223 

Description:
    Implementions of different attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial Attention module.

    Parameters
    ----------
    kernel_size : int, optional
        The size of the convolutional kernel. Default is 7.

    Attributes
    ----------
    conv1 : torch.nn.Conv2d
        Convolution layer with specified kernel size and padding.
    activate : torch.nn.Sigmoid
        Sigmoid activation function.

    References
    ----------
    .. [1] https://github.com/Jongchan/attention-module/tree/5d3a54af0f6688bedca3f179593dff8da63e8274
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.activate = nn.Sigmoid()

    def forward(self, x):  # [B,C,H,W]
        """
        Forward pass for the Spatial Attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Output tensor with shape [B, 1, H, W].
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]
        x = self.conv1(x)
        return self.activate(x)  # [B,1,H,W]


class ChannelAttention(nn.Module):
    """
    Channel Attention module.

    Parameters
    ----------
    channels : int
        The number of input channels.
    reduction_ratio : int, optional
        The reduction ratio for the MLP. Default is 16.

    Attributes
    ----------
    mlp : torch.nn.Sequential
        Multilayer perceptron consisting of Flatten, Linear, ReLU, and Linear layers.

    References
    ----------
    .. [1] https://github.com/Jongchan/attention-module/tree/5d3a54af0f6688bedca3f179593dff8da63e8274
    """

    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
        )

    def forward(self, x):
        """
        Forward pass for the Channel Attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Attention scaled tensor with the same shape as input [B, C, H, W].
        """
        h, w = x.shape[2:]
        avg_pool = F.avg_pool2d(x, (h, w), stride=(h, w))
        max_pool = F.max_pool2d(x, (h, w), stride=(h, w))
        channel_att = self.mlp(avg_pool) + self.mlp(max_pool)
        scale = F.sigmoid(channel_att).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Parameters
    ----------
    channels : int
        The number of input channels.
    reduction_ratio : int, optional
        The reduction ratio for the channel attention. Default is 16.
    spatial : bool, optional
        Whether to include spatial attention. Default is True.

    Attributes
    ----------
    channel_attn : ChannelAttention
        Channel attention module.
    spatial_attn : SpatialAttention or None
        Spatial attention module, if spatial is True; otherwise, None.

    References
    ----------
    .. [1] https://github.com/Jongchan/attention-module/tree/5d3a54af0f6688bedca3f179593dff8da63e8274
    """

    def __init__(self, channels, reduction_ratio=16, spatial=True):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(channels, reduction_ratio)
        if spatial:
            self.spatial_attn = SpatialAttention(kernel_size=7)
        else:
            self.spatial_attn = None

    def forward(self, x):
        """
        Forward pass for the CBAM module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input [B, C, H, W].
        """
        out = self.channel_attn(x)
        if self.spatial_attn:
            out = self.spatial_attn(out * x)
        return out


class TemporalAttention(nn.Module):
    """
    Self-attention module along the time dimension.

    Parameters
    ----------
    input_dim : int
        The number of input channels.
    hidden_dim : int
        The number of hidden channels for the query and key convolutions.
    out_dim : int
        The number of output channels for the value convolution.

    Attributes
    ----------
    query : torch.nn.Conv2d
        Convolutional layer for computing the query.
    key : torch.nn.Conv2d
        Convolutional layer for computing the key.
    value : torch.nn.Conv2d
        Convolutional layer for computing the value.
    softmax : torch.nn.Softmax
        Softmax layer for computing attention weights.

    References
    ----------
    .. [1] https://chatgpt.com/share/2233ba10-c2d1-4b06-9e83-41354cbfc1fe
    """

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.key = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(input_dim, out_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass for the Temporal Attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, T, C, H, W], where B is the batch size,
            T is the number of time steps, C is the number of channels, and H and W
            are the height and width of the feature maps.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input [B, T, out_dim, H, W].
        """
        b, t, c, h, w = x.shape
        # reshape for attention mechanism
        x = x.reshape((-1, c, h, w))  # [B*t,C,H,W]
        q = self.query(x).reshape(b, t, -1)  # [B*t,hidden,H,W] -> [B,t,hidden*H*W]
        k = self.key(x).reshape(b, t, -1)
        v = self.value(x).reshape(b, t, -1)  # [B,t,C*H*W]
        # compute attention:
        # Calculate the dot product of q and k to obtain the attention-fraction matrix.
        # This matrix represents the relationship between each time step and the other time steps.
        attention_weights = torch.bmm(q, k.permute(0, 2, 1))  # [B, t, t]
        attention_weights = self.softmax(attention_weights)
        # Calculate the weighted sum of v using attention_weights
        out = torch.bmm(attention_weights, v)  # [B, t, out_dim*H*W]
        out = out.reshape((b, t, -1, h, w))
        return out
