"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn


class TCCLIP_VE(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.dtype = torch.float32

        self.return_layer_num = [11]

    def forward(self, image, return_attention=False, return_source=False):
        # Encode visual features
        image_features, context_tokens, attn, source = self.image_encoder(image.type(self.dtype),
                                                                          return_layer_num=self.return_layer_num,
                                                                          return_attention=return_attention,
                                                                          return_source=return_source)

        # Now take the mean along the temporal direction with last layer cls tokens
        image_features_mean = image_features[:, -1, ...].mean(dim=1, keepdim=False)  # [b, 512]
        image_features = image_features.squeeze(1) # [b, frame_num, 512]
        return image_features_mean, image_features
