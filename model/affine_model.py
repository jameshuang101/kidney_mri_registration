import torch
import torch.nn as nn
from typing import Optional

class AffineModel(nn.Module):
    """
    Simple affine registration network that predicts 3×4 affine matrix.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(32, 12)  # 12 parameters for 3×4

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        """
        Predict affine parameters from moving & fixed volumes.

        Args:
            moving (Tensor): (B,1,D,H,W)
            fixed (Tensor): (B,1,D,H,W)

        Returns:
            theta (Tensor): (B,3,4) affine matrices
        """
        x = torch.cat([moving, fixed], dim=1)  # (B,2,D,H,W)
        features = self.encoder(x).view(x.size(0), -1)
        params = self.fc(features)
        theta = params.view(-1, 3, 4)
        return theta

    @classmethod
    def load_pretrained(cls, path: str, map_location: Optional[str]='cpu'):
        model = cls()
        model.load_state_dict(torch.load(path, map_location=map_location))
        model.eval()
        return model
