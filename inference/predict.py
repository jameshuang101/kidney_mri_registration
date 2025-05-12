import argparse
import torch
import torch.nn.functional as F
from data.dicom_loader import load_dicom_series
from models.affine_model import AffineModel
from models.deformable_model import DeformableModel
from utils.config import DEVICE
import numpy as np

def predict_pipeline(vol: np.ndarray, affine_path: str, deformable_path: str) -> np.ndarray:
    """
    Apply affine and deformable registration in sequence.

    Args:
        vol (np.ndarray): (D,H,W)
        affine_path (str): path to affine .pth
        deformable_path (str): path to deformable .pth
    Returns:
        registered (np.ndarray): (D,H,W)
    """
    model_a = AffineModel.load_pretrained(affine_path).to(DEVICE)
    model_d = DeformableModel.load_pretrained(deformable_path).to(DEVICE)

    # pick reference index (middle)
    ref_idx = vol.shape[0] // 2
    fixed = torch.from_numpy(vol[ref_idx][None,None]).to(DEVICE)

    registered = np.zeros_like(vol)
    registered[ref_idx] = vol[ref_idx]

    for i in range(vol.shape[0]):
        moving = torch.from_numpy(vol[i][None,None]).to(DEVICE)
        theta = model_a(moving, fixed)
        grid = F.affine_grid(theta, fixed.shape, align_corners=False)
        warped_a = F.grid_sample(moving, grid, align_corners=False)

        disp = model_d(warped_a, fixed)
        coords = make_grid(*fixed.shape[-3:], device=DEVICE) + disp.permute(0,2,3,4,1)
        warped = F.grid_sample(warped_a, coords, align_corners=False)

        registered[i] = warped.cpu().numpy()[0,0]
    return registered

def make_grid(D, H, W, device):
    zs = torch.linspace(-1,1,D, device=device)
    ys = torch.linspace(-1,1,H, device=device)
    xs = torch.linspace(-1,1,W, device=device)
    grid = torch.stack(torch.meshgrid(zs, ys, xs), -1)  # (D,H,W,3)
    return grid.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description='Run full registration pipeline.')
    parser.add_argument('--input', required=True, help='DICOM folder')
    parser.add_argument('--affine', required=True)
    parser.add_argument('--deformable', required=True)
    parser.add_argument('--output', required=True, help='Output .npy file')
    args = parser.parse_args()

    vol, meta = load_dicom_series(args.input)
    reg = predict_pipeline(vol, args.affine, args.deformable)
    np.save(args.output, reg)
    print(f'Registered volume saved to {args.output}')

if __name__ == '__main__':
    main()
