"""
lora_demo.py - WHY LoRA + quantization work, in pure NumPy (runnable, no GPU).

    python -m src.lora_demo

WHAT : demonstrates (1) LoRA's huge parameter saving, (2) that weight updates are approximately
       low-rank (so a small adapter suffices), and (3) 4-bit quantization's memory win - with
       assertions so the claims are verified.
WHY  : these two tricks are what let you fine-tune a billion-parameter model on ONE GPU (QLoRA).
HOW  : count params; approximate a matrix with a low rank via SVD; quantize floats to 16 levels.
WHERE: the concept layer. qlora_finetune.py is production (PEFT/TRL on a real model).
       See notebooks/08_lora_qlora_finetuning.ipynb for the rendered diagrams.
"""
from __future__ import annotations
import numpy as np


def lora_param_counts(d: int = 4096):
    """Full fine-tune trains d*d numbers; LoRA trains 2*d*r. Returns (full, {r: lora_count})."""
    full = d * d
    lora = {r: 2 * d * r for r in (4, 8, 16, 32)}
    return full, lora


def low_rank(M, r):
    """Best rank-r approximation of M via SVD (this is what a LoRA adapter approximates)."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vt[:r]


def quantize(w, bits: int = 4):
    """Map floats onto 2**bits evenly spaced levels (symmetric min-max), return dequantized + scale."""
    qmax = 2 ** bits - 1
    scale = (w.max() - w.min()) / qmax
    q = np.round((w - w.min()) / scale)
    return q * scale + w.min(), scale


def main():
    rng = np.random.default_rng(0)
    d = 4096

    full, lora = lora_param_counts(d)
    print(f"[lora] full fine-tune: {full:,} params/matrix")
    for r, n in lora.items():
        print(f"[lora] LoRA r={r:>2}: {n:,} params  ({full // n}x fewer)")
    assert lora[8] * 100 < full, "LoRA must train far fewer parameters than full fine-tuning"

    # updates are approximately low-rank: a rank-5-ish matrix is well approximated by small r
    target = rng.standard_normal((64, 5)) @ rng.standard_normal((5, 64)) + 0.1 * rng.standard_normal((64, 64))
    err = np.linalg.norm(target - low_rank(target, 5)) / np.linalg.norm(target)
    print(f"[lora] rank-5 reconstruction error: {err:.3f}")
    assert err < 0.2, "a low-rank patch should capture most of the update"

    # 4-bit quantization: 8x less memory than fp32
    W = rng.standard_normal(20000).astype(np.float32)
    deq, _ = quantize(W, bits=4)
    mem_ratio = (W.nbytes) / (len(W) * 4 // 8)
    print(f"[quant] 4-bit memory saving: {mem_ratio:.0f}x | avg error {np.abs(W - deq).mean():.4f}")
    assert mem_ratio >= 7.9, "4-bit should be ~8x smaller than fp32"
    print("[ok] LoRA + quantization properties verified")


if __name__ == "__main__":
    main()
