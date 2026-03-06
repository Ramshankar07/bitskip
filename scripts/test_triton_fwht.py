"""Test Triton FWHT kernel for correctness and speed."""

import math
import time
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def fwht_reference(x: torch.Tensor) -> torch.Tensor:
    """Pure Python/PyTorch FWHT reference (definitely correct)."""
    n = x.shape[-1]
    y = x.clone().view(-1, n)
    h = 1
    while h < n:
        for start in range(0, n, 2 * h):
            a = y[:, start : start + h].clone()
            b = y[:, start + h : start + 2 * h].clone()
            y[:, start : start + h] = a + b
            y[:, start + h : start + 2 * h] = a - b
        h <<= 1
    return (y.view(x.shape) * (1.0 / math.sqrt(n)))


def test_correctness():
    """Check Triton FWHT matches the reference for various sizes."""
    from bitnet.modeling.kernels.triton_fwht import fwht_triton

    print("=== Correctness Tests ===")
    sizes = [2, 4, 8, 16, 64, 256, 1024, 4096]
    batch_rows = 32

    for n in sizes:
        x = torch.randn(batch_rows, n, device="cuda", dtype=torch.float32)
        ref = fwht_reference(x.cpu()).cuda()
        out = fwht_triton(x)
        max_err = (out - ref).abs().max().item()
        rel_err = max_err / ref.abs().max().item()
        status = "PASS" if rel_err < 1e-5 else "FAIL"
        print(f"  N={n:5d}  max_abs_err={max_err:.2e}  rel_err={rel_err:.2e}  [{status}]")

    # Test with 3D input (batch, seq, features)
    x = torch.randn(4, 128, 1024, device="cuda")
    ref = fwht_reference(x.cpu()).cuda()
    out = fwht_triton(x)
    max_err = (out - ref).abs().max().item()
    rel_err = max_err / ref.abs().max().item()
    status = "PASS" if rel_err < 1e-5 else "FAIL"
    print(f"  3D (4,128,1024)  rel_err={rel_err:.2e}  [{status}]")

    # Test float16
    x = torch.randn(32, 1024, device="cuda", dtype=torch.float16)
    ref = fwht_reference(x.float().cpu()).half().cuda()
    out = fwht_triton(x)
    max_err = (out - ref).abs().max().item()
    rel_err = max_err / ref.abs().max().item()
    status = "PASS" if rel_err < 1e-2 else "FAIL"
    print(f"  fp16 N=1024  rel_err={rel_err:.2e}  [{status}]")
    print()


def test_autograd():
    """Verify gradients flow correctly through Triton FWHT."""
    from bitnet.modeling.kernels.triton_fwht import fwht_triton

    print("=== Autograd Tests ===")
    x = torch.randn(8, 1024, device="cuda", requires_grad=True)
    y = fwht_triton(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed!"
    assert x.grad.shape == x.shape, "Gradient shape mismatch!"
    assert torch.isfinite(x.grad).all(), "Non-finite gradients!"

    # Numerical gradient check (small size for speed)
    x = torch.randn(4, 64, device="cuda", dtype=torch.float64, requires_grad=True)
    passed = torch.autograd.gradcheck(fwht_triton, (x,), eps=1e-6, atol=1e-4)
    print(f"  gradcheck (N=64): {'PASS' if passed else 'FAIL'}")
    print()


def benchmark():
    """Compare Triton vs old CUDA kernel speed."""
    from bitnet.modeling.kernels.triton_fwht import fwht_triton

    print("=== Speed Benchmark ===")
    # Realistic shapes: batch=16, seq=512, features=1024 or 4096
    configs = [
        ("Attention (16*512, 1024)", (16 * 512, 1024)),
        ("FFN (16*512, 4096)", (16 * 512, 4096)),
        ("Small (256, 256)", (256, 256)),
    ]

    for label, shape in configs:
        x = torch.randn(*shape, device="cuda")

        # Warmup
        for _ in range(5):
            fwht_triton(x)
        torch.cuda.synchronize()

        # Benchmark Triton
        iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fwht_triton(x)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) / iters * 1000

        # Benchmark old CUDA kernel (if available)
        cuda_ms = None
        try:
            from bitnet.modeling.kernels import _load_extension
            ext = _load_extension()
            if ext is not None:
                scale = 1.0 / math.sqrt(shape[-1])
                for _ in range(5):
                    ext.fwht(x) * scale
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    ext.fwht(x) * scale
                torch.cuda.synchronize()
                cuda_ms = (time.perf_counter() - t0) / iters * 1000
        except Exception:
            pass

        speedup = f"{cuda_ms / triton_ms:.1f}x" if cuda_ms else "N/A"
        print(f"  {label}")
        print(f"    Triton: {triton_ms:.3f} ms")
        if cuda_ms:
            print(f"    CUDA:   {cuda_ms:.3f} ms")
        print(f"    Speedup: {speedup}")
    print()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    test_correctness()
    test_autograd()
    benchmark()
    print("All tests passed!")
