from __future__ import annotations

import torch
import torch.nn as nn

import open_clip


class ImageEncoder(nn.Module):

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
        device="cpu",
    ):
        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        self.device = torch.device(device)

        self.model = self.model.to(self.device)

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        images,
    ):

        with torch.no_grad():

            features = self.model.encode_image(
                images
            )

        return features.float()