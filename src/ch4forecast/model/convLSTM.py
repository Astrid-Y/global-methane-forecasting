"""
Author: Boyang Hu
GitHub Username: edsml-bh223 

Description:
    Implementions of ConvLSTM backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ch4forecast.model.attention import SpatialAttention, CBAM, TemporalAttention


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell with optional attention mechanism.

    Parameters
    ----------
    input_dim : int
        The number of input channels.
    hidden_dim : int
        The number of hidden channels.
    kernel_size : tuple of int
        The size of the convolutional kernel.
    bias : bool
        If True, adds a learnable bias to the output.
    attention : str, optional
        Type of attention mechanism to use. Options are "CS" (Channel and Spatial Attention),
        "SA" (Spatial Attention), "CA" (Channel Attention), or None for no attention. Default is None.

    Attributes
    ----------
    hidden_dim : int
        The number of hidden channels.
    conv : torch.nn.Conv2d
        Convolutional layer for computing the gates.
    ci : torch.nn.Parameter
        Parameter for input gate.
    cf : torch.nn.Parameter
        Parameter for forget gate.
    co : torch.nn.Parameter
        Parameter for output gate.
    attention : torch.nn.Module or None
        Attention module if specified, otherwise None.

    References
    ----------
    .. [1] https://github.com/ndrplz/ConvLSTM_pytorch/tree/master
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, attention=None):
        super(ConvLSTMCell, self).__init__()

        self.hidden_dim = hidden_dim

        padding = tuple([k // 2 for k in kernel_size])

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # 4 gates
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        self.ci = nn.Parameter(torch.randn(1, hidden_dim, 1, 1))
        self.cf = nn.Parameter(torch.randn(1, hidden_dim, 1, 1))
        self.co = nn.Parameter(torch.randn(1, hidden_dim, 1, 1))

        if attention == "CS":
            self.attention = CBAM(input_dim + hidden_dim)
        elif attention == "SA":
            self.attention = SpatialAttention()
        elif attention == "CA":
            self.attention = CBAM(input_dim + hidden_dim, spatial=False)
        else:
            self.attention = None

    def forward(self, input_tensor, cur_state):
        """
        Forward pass for the ConvLSTMCell.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor with shape [B, input_dim, H, W].
        cur_state : tuple of torch.Tensor
            Current hidden and cell states, each with shape [B, hidden_dim, H, W].

        Returns
        -------
        tuple of torch.Tensor
            Next hidden and cell states, each with shape [B, hidden_dim, H, W].
        """
        # current hidden state and cell state
        h_cur, c_cur = cur_state  # [B, hidden_dim, H, W]
        combined = torch.cat([input_tensor, h_cur], dim=1)  # [B, hidden+input, H, W]

        # Apply spatial attention
        if self.attention:
            attention_map = self.attention(combined)
            combined = combined * attention_map

        combined_conv = self.conv(combined)  # [B, 4*hidden_dim, H, W]
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = F.sigmoid(cc_i + self.ci * c_cur)
        f = F.sigmoid(cc_f + self.cf * c_cur)
        g = F.tanh(cc_g)
        c_next = f * c_cur + i * g
        o = F.sigmoid(cc_o + self.co * c_next)
        h_next = o * F.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, field_size):
        """
        Initialize the hidden and cell states.

        Parameters
        ----------
        batch_size : int
            The batch size.
        field_size : tuple of int
            The height and width of the hidden states.

        Returns
        -------
        tuple of torch.Tensor
            Initialized hidden and cell states, each with shape [B, hidden_dim, H, W].
        """
        height, width = field_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM network consisting of multiple ConvLSTMCells.

    Parameters
    ----------
    input_dim : int
        The number of input channels.
    hidden_dim : int or list of int
        The number of hidden channels for each layer.
    kernel_size : tuple of int or list of tuple of int
        The size of the convolutional kernel for each layer.
    num_layers : int
        The number of ConvLSTM layers.
    batch_first : bool, optional
        If True, the input and output tensors are provided as [B, time, C, H, W].
        Otherwise, the shape should be [time, B, C, H, W]. Default is True.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    attention : str, optional
        Type of attention mechanism to use. Options are "CS" (Channel and Spatial Attention),
        "SA" (Spatial Attention), "CA" (Channel Attention), or None for no attention. Default is None.
    return_all_layer : bool, optional
        If True, return the output and states of all layers. Otherwise, return only the last layer.
        Default is False.

    Attributes
    ----------
    num_layers : int
        The number of ConvLSTM layers.
    batch_first : bool
        If True, the input and output tensors are provided as [B, time, C, H, W].
    return_all_layer : bool
        If True, return the output and states of all layers.
    cell_list : torch.nn.ModuleList
        List containing ConvLSTMCell modules.

    References
    ----------
    .. [1] https://github.com/ndrplz/ConvLSTM_pytorch/tree/master
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=True,  # [B, time, C, H, W] or [time, B, C, H, W]
        bias=True,
        attention=None,
        return_all_layer=False,
    ):
        super(ConvLSTM, self).__init__()

        hidden_dim, kernel_size = self._check_consistency(
            hidden_dim, kernel_size, num_layers
        )

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layer = return_all_layer

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dim[i],
                    kernel_size=kernel_size[i],
                    bias=bias,
                    attention=attention,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass for the ConvLSTM network.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor with shape [B, time, C, H, W] or [time, B, C, H, W].
        hidden_state : list of tuple of torch.Tensor, optional
            Initial hidden and cell states for each layer. If not provided, they are initialized to zeros.

        Returns
        -------
        tuple
            - layer_output_list: list of torch.Tensor
                List of output tensors for each layer, each with shape [B, time, hidden_dim, H, W].
            - last_state_list: list of tuple of torch.Tensor
                List of the last hidden and cell states for each layer, each with shape [B, hidden_dim, H, W].
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)  # [B, time, C, H, W]

        b, _, _, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, field_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)  # time steps
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  # [B,hidden_dim,H,W]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)
            # stack output of each time step as input of next layer
            layer_output = torch.stack(output_inner, dim=1)  # [B,t,hidden_dim,H,W]
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # only return output and state of the last layer
        if self.return_all_layer == False:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, field_size):
        """
        Initialize the hidden and cell states.

        Parameters
        ----------
        batch_size : int
            The batch size.
        field_size : tuple of int
            The height and width of the hidden states.

        Returns
        -------
        list of tuple of torch.Tensor
            Initialized hidden and cell states for each layer.
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, field_size))
        return init_states

    def _check_consistency(self, hidden_dim, kernel_size, num_layers):
        """
        Check and ensure consistency of hidden_dim and kernel_size.

        Parameters
        ----------
        hidden_dim : int, list of int, or tuple of int
            The number of hidden channels.
        kernel_size : tuple of int, list of tuple of int
            The size of the convolutional kernel.
        num_layers : int
            The number of layers.

        Returns
        -------
        tuple
            - hidden_dim: list of int
                The number of hidden channels for each layer.
            - kernel_size: list of tuple of int
                The size of the convolutional kernel for each layer.

        Raises
        ------
        ValueError
            If hidden_dim or kernel_size are not of expected types or lengths.
        """
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        elif isinstance(hidden_dim, list) or isinstance(hidden_dim, tuple):
            if len(hidden_dim) != num_layers:
                raise ValueError("`hidden_dim` length must be equal to `num_layers`")
        else:
            raise ValueError("`hidden_dim` must be int or tuple or list")

        if isinstance(kernel_size, tuple):
            kernel_size = [kernel_size] * num_layers
        elif isinstance(kernel_size, list) and all(
            isinstance(elem, tuple) for elem in kernel_size
        ):
            if len(kernel_size) != num_layers:
                raise ValueError("`kernel_size` length must be equal to `num_layers`")
        else:
            raise ValueError("`kernel_size` must be tuple or list of tuples")
        return hidden_dim, kernel_size


class CALSTMNet(nn.Module):
    """
    Attention-based convolutional LSTM Network.

    Parameters
    ----------
    input_dim : int
        The number of input channels for the ConvLSTM.
    hidden_dim : int or list of int
        The number of hidden channels for each ConvLSTM layer.
    kernel_size : tuple of int or list of tuple of int
        The size of the convolutional kernel for each ConvLSTM layer.
    num_layers : int
        The number of ConvLSTM layers.
    output_dim : int
        The number of output channels.
    attention : str, optional
        Type of attention mechanism to use in ConvLSTM. Options are "CS" (Channel and Spatial Attention),
        "SA" (Spatial Attention), "CA" (Channel Attention), or None for no attention.

    Attributes
    ----------
    lstm : ConvLSTM
        Convolutional LSTM module for sequence processing.
    conv : torch.nn.Conv2d
        Convolutional layer for generating the final output.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        output_dim,
        attention,
    ):
        super(CALSTMNet, self).__init__()

        self.lstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            attention=attention,
        )

        self.conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for the CALSTMNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the ConvLSTM with shape [B, t, C, H, W].

        Returns
        -------
        torch.Tensor
            Output tensor with shape [B, t, output_dim, H, W].
        """
        # ConvLSTM to predict ch4
        lstm_out, _ = self.lstm(x)  # [B,t,hidden_dim,H,W]
        out_shape = lstm_out.shape
        out = self.conv(lstm_out.reshape(-1, *out_shape[2:]))  # convert to 4 dims
        out = F.sigmoid(out)
        out = out.reshape(*out_shape[:2], -1, *out_shape[3:])  # [B,t,1,H,W]
        return out


class TALSTMNet(nn.Module):
    """
    Multiple attention-based convolutional LSTM Network.

    Parameters
    ----------
    input_dim : int
        The number of input channels for the ConvLSTM.
    hidden_dim : int or list of int
        The number of hidden channels for each ConvLSTM layer.
    kernel_size : tuple of int or list of tuple of int
        The size of the convolutional kernel for each ConvLSTM layer.
    num_layers : int
        The number of ConvLSTM layers.
    output_dim : int
        The number of output channels.
    attention : str, optional
        Type of attention mechanism to use in ConvLSTM. Options are "CS" (Channel and Spatial Attention),
        "SA" (Spatial Attention), "CA" (Channel Attention), or None for no attention. Default is None.

    Attributes
    ----------
    lstm : ConvLSTM
        Convolutional LSTM module for sequence processing.
    att_layer : TemporalAttention
        Self-attention layer (along the time dimension) for enhancing time dependencies.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        output_dim,
        attention,
    ):
        super(TALSTMNet, self).__init__()

        self.lstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            attention=attention,
        )

        self.att_layer = TemporalAttention(hidden_dim, hidden_dim // 8, output_dim)

    def forward(self, x):
        """
        Forward pass for the TALSTMNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the ConvLSTM with shape [B, t, C, H, W].

        Returns
        -------
        torch.Tensor
            Output tensor with shape [B, t, output_dim, H, W].
        """
        # ConvLSTM to predict ch4
        lstm_out, _ = self.lstm(x)  # [B,t,hidden_dim,H,W]
        att_out = self.att_layer(lstm_out)
        out = F.sigmoid(att_out)
        return out
