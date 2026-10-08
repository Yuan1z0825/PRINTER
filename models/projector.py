import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Linear(input_nc, output_nc),
                nn.ReLU(),
                nn.Linear(output_nc, output_nc),
            ]
        )

    def forward(self, x):
        return self.mlp(x)


class Head(nn.Module):
    def __init__(self, in_channels=3, features=64, residuals=9):
        super().__init__()
        self.mlp_0 = MLP(3, 256)
        self.mlp_1 = MLP(128, 256)
        self.mlp_2 = MLP(256, 256)
        self.mlp_3 = MLP(256, 256)
        self.mlp_4 = MLP(256, 256)

    def forward(self, features):
        return_features = []
        for feature_id, feature in enumerate(features):
            mlp = getattr(self, f"mlp_{feature_id}")
            feature = mlp(feature)
            norm = feature.pow(2).sum(1, keepdim=True).pow(1.0 / 2)
            feature = feature.div(norm + 1e-7)
            return_features.append(feature)
        return return_features
