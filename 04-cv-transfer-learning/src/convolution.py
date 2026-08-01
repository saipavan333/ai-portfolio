"""
convolution.py - the heart of computer vision, in pure NumPy (runnable, no GPU).

    python -m src.convolution

WHAT : implements 2-D convolution and applies edge/blur filters to a synthetic image,
       proving (with assertions) that an edge filter responds at edges and a blur smooths.
WHY  : a CNN is just many *learned* convolution filters stacked up. Build the operation once
       by hand and the rest of vision (CNNs, ViTs, transfer learning) becomes legible.
HOW  : make an image -> slide each kernel over it (weighted sum per patch) -> a 'feature map'.
WHERE: the concept layer. transfer_learning.py is the production layer (real pretrained CNN).
       See notebooks/04_computer_vision.ipynb for the rendered diagrams.
"""
from __future__ import annotations
import numpy as np


def make_image(size: int = 40) -> np.ndarray:
    """A simple grayscale picture: a bright square plus a diagonal line (clear edges)."""
    img = np.zeros((size, size))
    img[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 1.0   # filled square
    np.fill_diagonal(img, 0.7)                                       # a diagonal line
    return img


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide `kernel` over `image`; each output pixel is the weighted sum of a patch.
    This single operation is what every convolutional layer computes."""
    kh, kw = kernel.shape
    out = np.zeros((image.shape[0] - kh + 1, image.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i:i + kh, j:j + kw]      # the little window under the kernel
            out[i, j] = float(np.sum(patch * kernel))   # element-wise multiply then sum
    return out


# Classic hand-designed filters (a CNN would LEARN these instead).
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   # responds to vertical edges
SOBEL_Y = SOBEL_X.T                                         # responds to horizontal edges
BLUR = np.ones((3, 3)) / 9.0                                # local average -> smoothing


def main() -> None:
    img = make_image()
    edges_x = np.abs(conv2d(img, SOBEL_X))
    edges_y = np.abs(conv2d(img, SOBEL_Y))
    blurred = conv2d(img, BLUR)

    print(f"[conv] image {img.shape} -> feature map {edges_x.shape}")
    print(f"[conv] vertical-edge response  : max={edges_x.max():.2f}")
    print(f"[conv] horizontal-edge response: max={edges_y.max():.2f}")
    print(f"[conv] blur keeps values in [0,1]: max={blurred.max():.2f}")

    # sanity checks: an edge filter must respond strongly somewhere; blur must not exceed input
    assert edges_x.max() > 1.0, "edge filter should produce a strong response at edges"
    assert blurred.max() <= img.max() + 1e-6, "blur should not amplify beyond the input"
    print("[conv] OK - convolution behaves as expected")
    # Extend: design your own 3x3 kernel and see what it detects.


if __name__ == "__main__":
    main()
