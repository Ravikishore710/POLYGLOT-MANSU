#!/usr/bin/env python3
"""
scripts/smoke_test_a100_bf16.py
Hardware Smoke-Test & Native BF16 Tensor Core Validation Script.
Verifies Ampere sm_80+ native BF16 capability before launching Stage B unlearning/evaluation.
Outputs structured JSON receipt and returns dictionary with GPU specifications,
throughput, autograd correctness, peak VRAM, runtime, and billing estimates.
"""

import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn

def run_smoke_test(output_path: Path = None) -> dict:
    t_start = time.time()
    print("=== [SMOKE TEST] Validating NVIDIA Ampere A100 Native BF16 Capability ===")
    
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available on this machine!")
        return {"status": "FAIL", "error": "CUDA not available"}
        
    torch.cuda.reset_peak_memory_stats()
    
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Device: {device_name}")
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    print(f"Total VRAM: {vram_gb:.2f} GB")
    
    # 1. Hardware Architecture Verification
    is_ampere_or_newer = (cap[0] >= 8)
    print(f"Native BF16 Tensor Core Hardware Architecture (sm_80+): {is_ampere_or_newer}")
    if not is_ampere_or_newer:
        print(f"FAIL: GPU architecture {cap} does not have native BF16 Tensor Cores!")
        return {"status": "FAIL", "error": f"sm_{cap[0]}{cap[1]} does not support native BF16"}
        
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"PyTorch torch.cuda.is_bf16_supported(): {bf16_supported}")
    assert bf16_supported, "FAIL: PyTorch reports BF16 is not supported on this CUDA driver!"

    # 2. Native BF16 GEMM Tensor Core Operation Test
    print("\n--- Step 1: Native BF16 Matrix Multiplication (GEMM) ---")
    M, K, N = 4096, 4096, 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(5):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark throughput
    t0 = time.time()
    iters = 20
    for _ in range(iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    t_gemm_elapsed = (time.time() - t0) / iters
    tflops = (2 * M * N * K) / (t_gemm_elapsed * 1e12)
    
    print(f"GEMM Shape: [{M}x{K}] @ [{K}x{N}] in BF16")
    print(f"Average GEMM Latency: {t_gemm_elapsed*1000:.2f} ms")
    print(f"Measured BF16 Tensor Core Throughput: {tflops:.2f} TFLOPS")
    assert C.dtype == torch.bfloat16, f"FAIL: Expected BF16 output, got {C.dtype}"
    assert not torch.isnan(C).any(), "FAIL: NaNs detected in BF16 GEMM output!"
    print("PASS: Native BF16 GEMM executed cleanly on Tensor Cores.")

    # 3. Model Forward and Backward Pass Smoke Test
    print("\n--- Step 2: Model Forward & Backward Pass Smoke Test in BF16 ---")
    layer = nn.Sequential(
        nn.Linear(2048, 8192, bias=False, dtype=torch.bfloat16),
        nn.GELU(),
        nn.Linear(8192, 2048, bias=False, dtype=torch.bfloat16)
    ).cuda()
    
    x = torch.randn(4, 512, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(4, 512, 2048, device="cuda", dtype=torch.bfloat16)
    
    out = layer(x)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
    
    assert layer[0].weight.grad is not None, "FAIL: Gradient not populated!"
    assert layer[0].weight.grad.dtype == torch.bfloat16, f"FAIL: Expected BF16 gradient, got {layer[0].weight.grad.dtype}"
    assert not torch.isnan(layer[0].weight.grad).any(), "FAIL: NaNs in gradient!"
    print(f"Forward-Backward Loss: {loss.item():.4f}")
    print(f"Layer weight gradient dtype: {layer[0].weight.grad.dtype}")
    print("PASS: Forward and backward passes completed natively in BF16.")

    # 4. Resource & Billing Tracking
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    runtime_sec = time.time() - t_start
    unit_cost_usd_per_min = 0.041667  # HF A100-large: $2.50/hr = $0.041667/min
    est_billing_cost_usd = (runtime_sec / 60.0) * unit_cost_usd_per_min

    results = {
        "status": "PASS",
        "gpu_model": device_name,
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "total_vram_gb": round(vram_gb, 2),
        "sm_80_plus": is_ampere_or_newer,
        "torch_bf16_supported": bf16_supported,
        "gemm_throughput_tflops": round(tflops, 2),
        "gemm_latency_ms": round(t_gemm_elapsed * 1000, 2),
        "gemm_output_dtype": str(C.dtype),
        "forward_backward_loss": round(loss.item(), 4),
        "weight_gradient_dtype": str(layer[0].weight.grad.dtype),
        "gradient_has_nan": bool(torch.isnan(layer[0].weight.grad).any().item()),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "runtime_seconds": round(runtime_sec, 2),
        "hf_hourly_rate_usd": 2.50,
        "estimated_cost_usd": round(est_billing_cost_usd, 5)
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSmoke test receipt saved to: {output_path}")

    print("\n=== [SMOKE TEST PASSED] Hardware is certified for Stage B BF16 execution ===")
    print(f"Peak VRAM: {peak_vram_gb:.4f} GB | Runtime: {runtime_sec:.2f}s | Est Cost: ${est_billing_cost_usd:.5f}")
    return results

def main():
    out_file = Path("results/smoke_test_a100_receipt.json")
    res = run_smoke_test(out_file)
    if res.get("status") != "PASS":
        sys.exit(1)

if __name__ == "__main__":
    main()
