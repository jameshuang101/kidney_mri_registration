import matplotlib.pyplot as plt
import numpy as np

def visualize_slice_pair(moving: np.ndarray, fixed: np.ndarray, registered: np.ndarray, idx: int):
    """
    Display moving, fixed, and registered slices side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    axes[0].imshow(moving[idx], cmap='gray'); axes[0].set_title('Moving')
    axes[1].imshow(fixed[idx], cmap='gray'); axes[1].set_title('Fixed')
    axes[2].imshow(registered[idx], cmap='gray'); axes[2].set_title('Registered')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
