# Implementation is heavily based on https://github.com/labmlai/annotated_deep_learning_paper_implementations
# with minor edits to make it a 1D model
from typing import Any, Optional, Dict, Union
from Diffusion236610.models.ae import Autoencoder
from Diffusion236610.models.unet import UNet1D, FixedUNet1D

import torch.nn as nn
import torch.nn.functional


class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model (LDM)

    This contains following components:
     -- AutoEncoder
     -- U-Net
     -- Domain-Specific conditioning module
    """

    def __init__(
            self,
            unet_model: Union[UNet1D, FixedUNet1D],
            autoencoder: Autoencoder,
            latent_scaling_factor: float,
            conditioning_module: Optional[nn.Module] = None,
            scale_latent_space: bool = True,
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """
        :param unet_model: is the U-Net that predicts noise $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param autoencoder: is the (pre-trained) AutoEncoder
        :param conditioning_module: is the domain-specific conditioning module
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        """

        super().__init__()

        # Set U-Net model
        self.unet = unet_model

        # Auto-encoder and scaling factor
        self.autoencoder = autoencoder
        self._scale_latent_space = scale_latent_space
        if scale_latent_space:
            self.latent_scaling_factor = latent_scaling_factor

        else:
            self.latent_scaling_factor = 1

        # domain-specific conditioning module
        self.cond_module = conditioning_module

        # Set device
        self._device = device

    @property
    def device(self):
        """
        Get model device
        """

        return self._device

    def get_conditioning(self, cond_input: Any):
        """
        Get the conditioning for the model
        """

        return self.cond_module(cond_input)

    def autoencoder_encode(self, x: torch.Tensor):
        """
        Get scaled latent space representation of the image
        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """

        return self.latent_scaling_factor * self.autoencoder.encode(x)

    def autoencoder_decode(self, z: torch.Tensor):
        """
         Get image from the latent representation
        We scale down by the scaling factor and then decode.
        """

        return self.autoencoder.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.
        $$\epsilon_\text{cond}(x_t, c)$$
        """

        cond = self.get_conditioning(context)
        out = self.unet(x, t, cond)

        return out

    def __call__(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        return self.forward(x=x, t=t, context=context)
