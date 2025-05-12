import argparse
import os
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
from data.dicom_loader import load_dicom_series
from models.deformable_model import DeformableModel
from utils.config import DEVICE
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

class MRIDataset(Dataset):
    def __init__(self, root_dir):
        self.cases = sorted(os.listdir(root_dir))
        self.root = root_dir

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        vol, _ = load_dicom_series(os.path.join(self.root, case))
        i, j = np.random.choice(vol.shape[0], 2, replace=False)
        moving = vol[i][None]
        fixed = vol[j][None]
        return torch.from_numpy(moving), torch.from_numpy(fixed)

def main():
    parser = argparse.ArgumentParser(description='Train deformable registration model.')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', type=str, default='deformable.pth')
    args = parser.parse_args()

    dataset = MRIDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = DeformableModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total = 0
        for moving, fixed in tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            moving, fixed = moving.to(DEVICE), fixed.to(DEVICE)
            disp = model(moving, fixed)
            # warp
            B, _, D, H, W = disp.shape
            grid = F.interpolate(disp, size=(D, H, W), mode='trilinear', align_corners=False)
            # convert disp to sampling grid
            coords = make_grid(D, H, W, device=disp.device) + grid.permute(0,2,3,4,1)
            warped = F.grid_sample(moving, coords, align_corners=False)
            loss = loss_fn(warped, fixed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'Epoch {epoch+1} loss: {total/len(loader):.4f}')

    torch.save(model.state_dict(), args.out)
    print(f'Saved deformable model to {args.out}')

def make_grid(D, H, W, device):
    # create normalized meshgrid
    zs = torch.linspace(-1,1,D, device=device)
    ys = torch.linspace(-1,1,H, device=device)
    xs = torch.linspace(-1,1,W, device=device)
    grid = torch.stack(torch.meshgrid(zs, ys, xs), -1)  # (D,H,W,3)
    return grid.unsqueeze(0).repeat(1,1,1,1,1)

if __name__ == '__main__':
    main()
