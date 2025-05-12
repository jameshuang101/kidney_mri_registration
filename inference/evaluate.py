import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from data.dicom_loader import load_dicom_series

def compute_mse(vol1: np.ndarray, vol2: np.ndarray) -> float:
    """
    Compute mean squared error between two volumes.
    """
    return mean_squared_error(vol1.ravel(), vol2.ravel())

def main():
    parser = argparse.ArgumentParser(description='Evaluate registration.')
    parser.add_argument('--fixed', required=True, help='Reference DICOM folder')
    parser.add_argument('--registered', required=True, help='Registered .npy file')
    args = parser.parse_args()

    fixed_vol, _ = load_dicom_series(args.fixed)
    reg_vol = np.load(args.registered)

    mse = compute_mse(fixed_vol, reg_vol)
    print(f'MSE: {mse:.6f}')

if __name__ == '__main__':
    main()
