from abc import ABC
from torch import Tensor
from typing import Union
from Diffusion236610.models.activations import NLSq, Snake, Swish
from Diffusion236610.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY, OTHER_KEY

import torch
import torch.nn as nn


def init_weights(module: nn.Module):
    """
    Utility method for initializing weights in layers

    :param module: (PyTorch Module) The layer to be initialized.

    :return: None
    """

    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)


def get_activation_layer(activation: Union[str, dict] = 'relu') -> nn.Module:
    """
    A utility method for defining and instantiating a PyTorch
    non-linear activation layer.

    :param activation: (str / dict / None) non-linear activation function to apply.
    If a string is given, using the layer with default parameters.
    if dict is given uses the 'name' key to determine which activation function to
    use and the 'params' key should have a dict with the required parameters as a
    key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
    'hardshrink', 'leakyrelu', 'prelu', 'tanh', 'snake', 'nlsq', default = 'relu'

    :return: (PyTorch Module) The activation layer.
    """

    # Define the activations keys-modules pairs
    activations_dict = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'elu': nn.ELU,
        'hardshrink': nn.Hardshrink,
        'leakyrelu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'tanh': nn.Tanh,
        'snake': Snake,
        'nlsq': NLSq,
        'swish': Swish,
        'sigmoid': nn.Sigmoid
    }

    # Validate inputs
    if (activation is not None and not isinstance(activation, str)
            and not isinstance(activation, dict)):
        raise ValueError("Can't take specs for the activation layer of type"
                         f" {type(activation)}, please specify either with a "
                         "string or a dictionary.")

    if (isinstance(activation, dict)
            and activation['name'] not in activations_dict.keys()):
        raise ValueError(f"{activation['name']} is not a supported Activation type.")

    if isinstance(activation, str):
        activation_block = activations_dict[activation]()

    else:
        activation_block = activations_dict[activation['name']](
            **activation['params'])

    return activation_block


def get_normalization_layer(norm: dict) -> nn.Module:
    """
    A utility method for defining and instantiating a PyTorch normalization layer.

    :param norm: (dict / None) Denotes the normalization layer to use with the FC layer.
    The dict should contains at least two keys, 'name' for indicating the type of
    normalization to use, and 'params', which should also map to a dict with all
    required parameters for the normalization layer. At the minimum, the 'params' dict
    should define the 'num_channels' key to indicate the expected number of
    channels on which to apply the normalization. For the GroupNorm, it is also
    required to specify a 'num_groups' key.
    If None then doesn't add normalization layer.
    Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
    'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
    BatchNorm, `instance` stands for InstanceNorm and `group` stands
    for GroupNorm. Default == None.

    :return: (PyTorch Module) The normalization layer.
    """

    # Define the normalizations keys-modules pairs
    norms_dict = {
        'batch1d': nn.BatchNorm1d,
        'batch2d': nn.BatchNorm2d,
        'batch3d': nn.BatchNorm3d,
        'instance1d': nn.InstanceNorm1d,
        'instance2d': nn.InstanceNorm2d,
        'instance3d': nn.InstanceNorm3d,
        'layer': nn.LayerNorm,
        'group': nn.GroupNorm,
    }

    # Validate inputs
    if norm is not None and not isinstance(norm, dict):
        raise ValueError(f"Can't specify norm as a {type(norm)} type. Please "
                         f"either use a dict, or None.")

    if (isinstance(norm, dict)
            and norm['name'] not in norms_dict.keys()):
        raise ValueError(f"{norm['name']} is not a supported Normalization type.")

    norm_block = norms_dict[norm['name']](
        **norm['params'])

    return norm_block


def get_fc_layer(
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        activation: Union[str, dict, None] = 'relu',
        dropout: Union[float, None] = None,
        norm: Union[dict, None] = None,
        mc_dropout: bool = False,
) -> nn.Module:
    """
    A utility method for generating a FC layer

    :param input_dim: (int) input dimension of the 2D matrices
    (M for a N X M matrix)
    :param output_dim: (int) output dimension of the 2D matrices
     (M for a N X M matrix)
    :param bias: (bool) whether to use a bias in the FC layer or not,
     default = False
    :param activation: (str / dict / None) non-linear activation function to apply.
    If a string is given, using the layer with default parameters.
    if dict is given uses the 'name' key to determine which activation function to
    use and the 'params' key should have a dict with the required parameters as a
    key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
    'hardshrink', 'leakyrelu', 'prelu', 'tanh', default = 'relu'
    :param dropout: (float/ None) rate of dropout to apply to the FC layer,
    if None than doesn't apply dropout, default = None
    :param norm: (dict / None) Denotes the normalization layer to use with the FC layer.
    The dict should contain at least two keys, 'name' for indicating the type of
    normalization to use, and 'params', which should also map to a dict with all
    required parameters for the normalization layer. At the minimum, the 'params' dict
    should define the 'num_channels' key to indicate the expected number of
    channels on which to apply the normalization. For the GroupNorm, it is also
    required to specify a 'num_groups' key.
    If None then doesn't add normalization layer.
    Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
    'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
    BatchNorm, `instance` stands for InstanceNorm and `group` stands
    for GroupNorm. Default == None.
    :param mc_dropout: (bool) Whether to use MC dropout or not

    :return: (PyTorch Module) the instantiated layer, according to the given specs
    """

    # Validate inputs
    if dropout is not None and (not isinstance(dropout, float) and not isinstance(dropout, int)):
        raise ValueError(f"Can't specify dropout as a {type(dropout)} type. Please "
                         f"either use float, or None.")

    # Add the FC block
    blocks = []
    fc_block = nn.Linear(
        in_features=input_dim,
        out_features=output_dim,
        bias=bias,
    )
    blocks.append(fc_block)

    # Add the Normalization block if required
    if norm is not None:
        norm_block = get_normalization_layer(norm)
        blocks.append(norm_block)

    # Add the Activation block if required
    if activation is not None:
        activation_block = get_activation_layer(activation)
        blocks.append(activation_block)

    # Add the Dropout block if required
    if dropout is not None:
        if mc_dropout:
            dropout_block = MCDropout(p=dropout)

        else:
            dropout_block = nn.Dropout(p=dropout)

        blocks.append(dropout_block)

    # Encapsulate all blocks as a single `Sequential` module.
    fc_layer = nn.Sequential(*blocks)

    return fc_layer


class SkipConnectionFCBlock(nn.Module):
    """
    A FC block which applies skip connection.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bias: bool = False,
            activation: Union[str, dict, None] = 'relu',
            dropout: Union[float, None] = None,
            norm: Union[dict, None] = None,
            mc_dropout: bool = False,
    ):
        super(SkipConnectionFCBlock, self).__init__()

        self._fc_block = get_fc_layer(
            input_dim=input_dim,
            output_dim=output_dim,
            bias=bias,
            activation=activation,
            dropout=dropout,
            norm=norm,
            mc_dropout=mc_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self._fc_block(x)
        out = out + x
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class MLP(nn.Module, ABC):
    """
    A general MLP class
    """

    def __init__(self,
                 n_layers: int,
                 in_channels: int,
                 out_channels: int,
                 l0_units: int = 1024,
                 units_grow_rate: int = 2,
                 grow_every_x_layer: int = 2,
                 bias: bool = False,
                 activation: Union[str, dict, None] = 'relu',
                 final_activation: Union[str, dict, None] = None,
                 norm: Union[dict, None] = None,
                 final_norm: Union[dict, None] = None,
                 dropout: Union[float, None] = None):
        """
        The constructor for the MLP class.

        :param n_layers: (int) Number of FC layers to include in the encoder.
        :param in_channels: (int) input dimension of the 2D matrices
        (M for a N X M matrix) to the MLP.
        :param out_channels: (int) output dimension of the 2D matrices
        (M for a N X M matrix) from the MLP.
        :param l0_units: (int) Number of units to include in the first FC layer.
        :param units_grow_rate: (int) The factor by which to increase the number of
        units between FC layers.
        :param grow_every_x_layer: (int) Indicate after how many layers to increase the
        number of units by a factor of 'grow_every_x_layer'.
        :param bias: (bool) whether to use a bias in the MLP layers or not,
        default = False.
        :param activation: (str / dict / None) The non-linear activation function
        to apply across all layers of the MLP, except for last one.
        If a string is given, using the layer with default parameters.
        if dict is given uses the 'name' key to determine which activation function to
        use and the 'params' key should have a dict with the required parameters as a
        key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
        'hardshrink', 'leakyrelu', 'prelu', 'tanh', default = 'relu'.
        :param final_activation: (str / dict / None) The non-linear activation function
        to apply only to the last layer. Takes in the same values as 'activation'.
        :param dropout: (float/ None) rate of dropout to apply across all MLP layers,
        if None than doesn't apply dropout, default = None.
        :param norm: (dict / None) Denotes the normalization layer to apply to all
         MLP layers, except for the last one.
        The dict should contains at least two keys, 'name' for indicating the type of
        normalization to use, and 'params', which should also map to a dict with all
        required parameters for the normalization layer. At the minimum, the 'params'
        dict should define the 'num_channels' key to indicate the expected number of
        channels on which to apply the normalization. For the GroupNorm, it is also
        required to specify a 'num_groups' key.
        If None then doesn't add normalization layer.
        Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
        'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
        BatchNorm, `instance` stands for InstanceNorm and `group` stands
        for GroupNorm. Default == None.
        :param final_norm: (dict / None) Denotes the normalization layer to apply only
        to the last MLP layer. The dict should contains at least two keys, 'name' for
        indicating the type of normalization to use, and 'params', which should also
        map to a dict with all required parameters for the normalization layer.
        At the minimum, the 'params' dict should define the 'num_channels' key to
        indicate the expected number of channels on which to apply the normalization.
        For the GroupNorm, it is also required to specify a 'num_groups' key.
        If None then doesn't add normalization layer.
        Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
        'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
        BatchNorm, `instance` stands for InstanceNorm and `group` stands
        for GroupNorm. Default == None.
        """

        super(MLP, self).__init__()

        # Define the first layer
        fc_layers = [
            get_fc_layer(
                input_dim=in_channels,
                output_dim=l0_units,
                bias=bias,
                activation=activation,
                dropout=dropout,
                norm=norm,
            ),
        ]

        # Define all layers except the first and last
        n_units = l0_units
        in_units = out_units = n_units
        for layer in range(1, (n_layers - 1)):

            # Increase the number of units every x layer
            if layer % grow_every_x_layer == 0:
                out_units *= units_grow_rate

            fc_layers.append(
                get_fc_layer(
                    input_dim=in_units,
                    output_dim=out_units,
                    bias=bias,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                )
            )

            in_units = out_units

        # Define the last layer
        fc_layers.append(
            get_fc_layer(
                input_dim=in_units,
                output_dim=out_channels,
                bias=bias,
                activation=final_activation,
                dropout=None,
                norm=final_norm,
            )
        )

        self._model = nn.Sequential(*fc_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward logic for the MLP model.

        :param x: (Tensor) contains the raw inputs.

        :return: (Tensor) the output of the forward pass of the MLP.
        """

        out = self._model(x)

        return out


class MCDropout(nn.Module):
    """
    A wrapper around PyTorch Dropout layer, which always uses the layer in 'train mode'
    """

    def __init__(self, p: float = 0.):
        """
        :param p: (float) rate of dropout. Defaults to 0.
        """

        super().__init__()

        self._dropout = nn.Dropout(p=p)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        self.train(True)
        return self._dropout(x)
