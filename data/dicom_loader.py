import os
import numpy as np
import pydicom
from typing import Tuple

def load_dicom_series(directory: str) -> Tuple[np.ndarray, dict]:
    """
    Load a DICOM series from a directory into a 3D numpy array.

    Args:
        directory (str): Path to folder containing DICOM (.dcm) files.

    Returns:
        volume (np.ndarray): 3D array of shape (D, H, W).
        metadata (dict): Common metadata from the first slice.
    """
    files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
    files.sort(key=lambda x: float(x.InstanceNumber))
    slices = [f.pixel_array for f in files]
    volume = np.stack(slices, axis=0)
    meta = {
        'pixel_spacing': files[0].PixelSpacing,
        'slice_thickness': files[0].SliceThickness
    }
    return volume.astype(np.float32), meta
