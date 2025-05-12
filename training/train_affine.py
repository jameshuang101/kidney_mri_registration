import argparse
import os
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
from data.dicom_loader import load_dicom_series
from models.affine_model import AffineModel
from utils.config import DEVICE
from tqdm import tqdm
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, root_dir):
        self.cases = sorted(os.listdir(root_dir))
        self.root = root_dir

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        vol, _ = load_dicom_series(os.path.join(self.root, case))
        # randomly pick two volumes as moving/fixed
        i, j = np.random.choice(vol.shape[0], 2, replace=False)
        moving = vol[i][None]
        fixed = vol[j][None]
        return torch.from_numpy(moving), torch.from_numpy(fixed)

def main():
    parser = argparse.ArgumentParser(description='Train affine registration model.')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', type=str, default='affine.pth')
    args = parser.parse_args()

    dataset = MRIDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = AffineModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total = 0
        for moving, fixed in tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            moving, fixed = moving.to(DEVICE), fixed.to(DEVICE)
            theta = model(moving, fixed)
            # apply affine transform
            grid = torch.nn.functional.affine_grid(theta, fixed.size(), align_corners=False)
            warped = torch.nn.functional.grid_sample(moving, grid, align_corners=False)
            loss = loss_fn(warped, fixed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'Epoch {epoch+1} loss: {total/len(loader):.4f}')

    torch.save(model.state_dict(), args.out)
    print(f'Saved affine model to {args.out}')

if __name__ == '__main__':
    main()
