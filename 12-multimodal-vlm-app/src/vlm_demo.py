"""
vlm_demo.py - how a vision-language model 'sees', in pure NumPy (runnable, no GPU).

    python -m src.vlm_demo

WHAT : turns an image into patch 'tokens' (like a ViT) and answers questions about it with a toy
       visual-Q&A function (dominant colour + shape) - so a VLM's input/output is concrete.
WHY  : multimodal (image + text) is a fast-growing 2026 specialisation; the key idea is that an
       image becomes a SEQUENCE of tokens, just like words, then a transformer attends across both.
HOW  : draw a shape -> split into patches -> average colour per patch (toy 'image tokens') -> answer.
WHERE: the concept layer. vlm_infer.py is production (a real open VLM).
       See notebooks/12_multimodal_vlm.ipynb for the patch-grid + VQA diagrams.
"""
from __future__ import annotations
import numpy as np


def make_image(shape="circle", color=(60, 90, 200), size=64):
    """A coloured shape on white, so we know the 'right answers' to test the toy VLM."""
    img = np.ones((size, size, 3)) * 255
    yy, xx = np.mgrid[0:size, 0:size]; cx = cy = size / 2
    if shape == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (size * 0.30) ** 2
    else:
        m = size * 0.28; mask = (np.abs(xx - cx) <= m) & (np.abs(yy - cy) <= m)
    for c in range(3):
        img[..., c][mask] = color[c]
    return img.astype(np.uint8), mask


def to_patch_tokens(img, patch=16):
    """A VLM splits the image into patches and treats each as a token. Toy token = patch's avg colour."""
    size = img.shape[0]
    grid = img.reshape(size // patch, patch, size // patch, patch, 3).mean(axis=(1, 3))
    return grid.reshape(-1, 3)        # (num_patches, 3): a sequence of 'image tokens'


def answer_color(img, mask):
    avg = img[mask].mean(0)
    names = {"red": (200, 40, 40), "green": (40, 160, 40), "blue": (60, 90, 200)}
    return min(names, key=lambda n: np.linalg.norm(avg - np.array(names[n])))


def answer_shape(mask):
    ys, xs = np.where(mask)
    fill = mask.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))   # square~1, circle~0.78
    return "square" if fill > 0.9 else "circle"


def main():
    img, mask = make_image("circle", (60, 90, 200))      # a blue circle
    tokens = to_patch_tokens(img)
    print(f"[vlm] image -> {tokens.shape[0]} patch tokens, each a vector of length {tokens.shape[1]}")
    color, shape = answer_color(img, mask), answer_shape(mask)
    print(f"[vlm] Q: what colour? A: {color}")
    print(f"[vlm] Q: what shape?  A: {shape}")
    assert color == "blue" and shape == "circle", "toy VQA should read a blue circle"
    print("[vlm] OK - image tokenised and questions answered correctly")


if __name__ == "__main__":
    main()
