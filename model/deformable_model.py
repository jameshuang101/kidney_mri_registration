import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DeformableModel(nn.Module):
    """
    Unet-style network that predicts a displacement field.
    """
    def __init__(self, in_channels=2, features=(16,32,64,32,16)):
        super().__init__()
        self.enc_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        # Encoder
        prev_ch = in_channels
        for ch in features[:3]:
            self.enc_convs.append(nn.Sequential(
                nn.Conv3d(prev_ch, ch, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2)))
            prev_ch = ch
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(prev_ch, features[2], 3, padding=1),
            nn.ReLU())
        # Decoder
        for ch in features[3:]:
            self.dec_convs.append(nn.Sequential(
                nn.ConvTranspose3d(prev_ch, ch, 2, stride=2),
                nn.ReLU()))
            prev_ch = ch
        # Final conv to 3 channels (dx,dy,dz)
        self.out_conv = nn.Conv3d(prev_ch, 3, 3, padding=1)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        """
        Predicts displacement field to warp moving to fixed.

        Args:
            moving, fixed: (B,1,D,H,W)
        Returns:
            disp: (B,3,D,H,W)
        """
        x = torch.cat([moving, fixed], dim=1)
        enc_feats = []
        for enc in self.enc_convs:
            x = enc(x)
            enc_feats.append(x)
        x = self.bottleneck(x)
        for dec, feat in zip(self.dec_convs, reversed(enc_feats)):
            x = dec(x)
            x = torch.cat([x, feat], dim=1)
        disp = self.out_conv(x)
        return disp

    @classmethod
    def load_pretrained(cls, path: str, map_location: Optional[str]='cpu'):
        model = cls()
        model.load_state_dict(torch.load(path, map_location=map_location))
        model.eval()
        return model
