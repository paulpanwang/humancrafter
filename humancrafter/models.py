from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


def _cfg(config, key, default):
    if config is None:
        return default
    return getattr(config, key, default)


if nn is None:

    class HumanCrafterModel:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch is required to initialize HumanCrafterModel")


    def load_checkpoint(model, checkpoint_path, map_location="cpu"):
        raise ModuleNotFoundError("torch is required to load HumanCrafter checkpoints")

else:

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

        def forward(self, inputs):
            return self.block(inputs)


    class HumanCrafterModel(nn.Module):
        """
        Lightweight baseline architecture that mirrors the public HumanCrafter task split:
        geometry, semantics, and appearance are predicted from a shared image encoder.
        """

        def __init__(self, model_config=None):
            super().__init__()

            hidden_dim = _cfg(model_config, "hidden_dim", 128)
            bottleneck_dim = _cfg(model_config, "bottleneck_dim", 256)
            feature_dim = _cfg(model_config, "feature_dim", 64)
            num_semantic_classes = _cfg(model_config, "num_semantic_classes", 4)

            self.encoder = nn.Sequential(
                ConvBlock(3, hidden_dim // 2, stride=2),
                ConvBlock(hidden_dim // 2, hidden_dim, stride=2),
                ConvBlock(hidden_dim, bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    bottleneck_dim, hidden_dim, kernel_size=4, stride=2, padding=1
                ),
                nn.GELU(),
                nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
            )
            self.point_head = nn.Conv2d(hidden_dim // 2, 3, kernel_size=1)
            self.semantic_head = nn.Conv2d(hidden_dim // 2, num_semantic_classes, kernel_size=1)
            self.appearance_head = nn.Conv2d(hidden_dim // 2, feature_dim, kernel_size=1)

        def forward(self, images):
            features = self.encoder(images)
            decoded = self.decoder(features)

            point_maps = self.point_head(decoded).permute(0, 2, 3, 1).contiguous()
            semantic_labels = self.semantic_head(decoded).permute(0, 2, 3, 1).contiguous()
            appearance_features = self.appearance_head(decoded).permute(0, 2, 3, 1).contiguous()

            return {
                "point_maps": point_maps,
                "semantic_labels": semantic_labels,
                "appearance_features": appearance_features,
            }


    def load_checkpoint(model, checkpoint_path, map_location="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        return model
