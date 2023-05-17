# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


from typing import List

# from ..modeling import Sam

class CoreMlModel(nn.Module):
    """
    This model should not be called directly, but is used in CoreML export.
    It combines the image preprocessing and image encoding of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the CoreML export script for details.
    """

    def __init__(
        self,
        longest_side: int = 1024,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.longest_side = longest_side
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    # Steps:
    # The dimensions of image encoder model is 1024x1024 for the b version
    # Image input in HWC uint8 format with pixel values in [0, 255] as RGB
    
    # Transform.apply image first calculates the preprocess shape (scales largest side to 1024 + rounds up to nearest int)
    # Then uses either F.Interpolate (torch) or NP -> PIL -> NP (regular) to set the image to that size and returns to predictor

    # Predictor converts the image to a tensor
    # Predictor then does this: input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] to swap dimension order 
    # and adds an extra dimension so it's compatible with the model (it expects a batch of imgs)

    # tensor is now 1x3xHxW with one of the H or W dimensions being 1024
    # Tensor and original img size (H, W) int are passed to be processed
    
    # The image is then run through Sam.preprocess, which normalizes the pixel values by hardcoded values and pads to a square input
    # The image is finally run through the image encoder, and added to the predictor as .features

    @torch.no_grad()
    def forward(
        self,
        input_image: torch.Tensor,
    ):
        # First, resize the image so the longest size is 1024
        # Get the input image size, which is in HWC format
        H, W, _ = input_image.shape    
        input_size = torch.tensor([H, W]).to(torch.float32)
        transformed_size = self.resize_longest_image_size(input_size, self.longest_side)

        # Change the image to BCHW format
        input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Resize the image to the transformed size
        input = F.interpolate(
            input_image,
            size=tuple(transformed_size.numpy().astype(int)),
            mode="bilinear",
            align_corners=False,
        )

        # Preprocess the image
        input = self.normalize_and_pad(input)

        return input

    def normalize_and_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.longest_side - h
        padw = self.longest_side - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def resize_longest_image_size(
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size 
    

