import argparse
from data.dicom_loader import load_dicom_series
from inference.predict import predict_pipeline
from utils.visualization import visualize_slice_pair

def main():
    parser = argparse.ArgumentParser(description='Full Kidney MRI Registration')
    parser.add_argument('--input', required=True, help='Input DICOM directory')
    parser.add_argument('--affine', required=True, help='Affine model .pth')
    parser.add_argument('--deformable', required=True, help='Deformable model .pth')
    parser.add_argument('--output', required=False, help='Optional: save registered .npy')
    parser.add_argument('--slice', type=int, default=None, help='Slice index to visualize')
    args = parser.parse_args()

    vol, _ = load_dicom_series(args.input)
    reg = predict_pipeline(vol, args.affine, args.deformable)

    if args.output:
        import numpy as np
        np.save(args.output, reg)
        print(f'Saved registered volume to {args.output}')

    idx = args.slice or vol.shape[0]//2
    visualize_slice_pair(vol, vol, reg, idx)

if __name__ == '__main__':
    main()
