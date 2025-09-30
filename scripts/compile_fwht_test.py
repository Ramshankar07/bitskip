import sys
import time
import torch

def main() -> int:
    print("=== FWHT CUDA Compile & Validation ===")
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    try:
        from bitnet.modeling import kernels
    except Exception as e:
        print("ERROR: Could not import kernels module:", e)
        return 2

    print("Kernels available before load:", kernels.is_available())

    # Choose device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Use power-of-two length
    n = 1024
    batch = 16

    # Random input
    x = torch.randn(batch, n, device=device, dtype=torch.float32)

    # Trigger build/load and time FWHT
    t0 = time.time()
    y = kernels.fwht(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    print(f"FWHT run OK. shape={tuple(y.shape)} time={t1-t0:.4f}s device={device}")
    print("Kernels available after load:", kernels.is_available())

    # Validate property: FWHT(FWHT(x)) == n * x (unnormalized)
    z = kernels.fwht(y)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    err = (z - n * x).abs().max().item()
    print(f"Validation max error vs n*x: {err:.3e}")

    # Simple threshold to pass
    if not torch.isfinite(torch.tensor(err)) or err > 1e-3:
        print("VALIDATION FAILED: error too high")
        return 1

    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


