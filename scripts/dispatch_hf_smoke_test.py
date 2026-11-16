#!/usr/bin/env python3
"""
scripts/dispatch_hf_smoke_test.py
Autonomous Hugging Face A100 Smoke Test Dispatcher & Safety Monitor.
1. Uploads smoke test code to private space Ravikishore710/polyglot-mansu-smoke-test.
2. Monitors space build and execution on Nvidia A100-80GB.
3. Streams logs until smoke test completes.
4. Extracts receipt and downloads smoke_test_receipt.json to results/.
5. IMMEDIATELY pauses space to stop GPU billing.
6. Reports full metrics, runtime, and billing details.
"""

import sys
import time
import json
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Ravikishore710/polyglot-mansu-smoke-test"

REQUIREMENTS_CONTENT = """torch
"""

APP_PY_CONTENT = """import sys
import time
import json
import gradio as gr
import torch
import torch.nn as nn

print("==================================================================", flush=True)
print("=== [SMOKE TEST] Validating NVIDIA Ampere A100 Native BF16 Capability ===", flush=True)
print("==================================================================", flush=True)

t_start = time.time()

if not torch.cuda.is_available():
    print("FATAL: CUDA is not available on this worker!", flush=True)
    results = {"status": "FAIL", "error": "CUDA not available"}
else:
    torch.cuda.reset_peak_memory_stats()
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Device: {device_name}", flush=True)
    print(f"Compute Capability: {cap[0]}.{cap[1]}", flush=True)
    print(f"Total VRAM: {vram_gb:.2f} GB", flush=True)
    
    is_ampere_or_newer = (cap[0] >= 8)
    print(f"Native BF16 Tensor Core Hardware Architecture (sm_80+): {is_ampere_or_newer}", flush=True)
    assert is_ampere_or_newer, f"sm_{cap[0]}.{cap[1]} does not have native BF16 Tensor Cores!"
    
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"PyTorch torch.cuda.is_bf16_supported(): {bf16_supported}", flush=True)
    assert bf16_supported, "PyTorch reports BF16 is not supported on this CUDA driver!"

    # 1. Native BF16 GEMM
    print("\\n--- Step 1: Native BF16 Matrix Multiplication (GEMM) ---", flush=True)
    M, K, N = 4096, 4096, 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    t0 = time.time()
    iters = 20
    for _ in range(iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    t_gemm = (time.time() - t0) / iters
    tflops = (2 * M * N * K) / (t_gemm * 1e12)
    
    print(f"GEMM Shape: [{M}x{K}] @ [{K}x{N}] in BF16", flush=True)
    print(f"Average GEMM Latency: {t_gemm*1000:.2f} ms", flush=True)
    print(f"Measured BF16 Tensor Core Throughput: {tflops:.2f} TFLOPS", flush=True)
    assert C.dtype == torch.bfloat16, f"Expected BF16 output, got {C.dtype}"
    assert not torch.isnan(C).any(), "NaNs detected in BF16 GEMM output!"
    print("PASS: Native BF16 GEMM executed cleanly on Tensor Cores.", flush=True)

    # 2. Forward and Backward autograd in BF16
    print("\\n--- Step 2: Model Forward & Backward Pass Smoke Test in BF16 ---", flush=True)
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
    
    assert layer[0].weight.grad is not None, "Gradient not populated!"
    assert layer[0].weight.grad.dtype == torch.bfloat16, f"Expected BF16 gradient, got {layer[0].weight.grad.dtype}"
    assert not torch.isnan(layer[0].weight.grad).any(), "NaNs in gradient!"
    print(f"Forward-Backward Loss: {loss.item():.4f}", flush=True)
    print(f"Layer weight gradient dtype: {layer[0].weight.grad.dtype}", flush=True)
    print("PASS: Forward and backward passes completed natively in BF16.", flush=True)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    runtime_sec = time.time() - t_start
    
    results = {
        "status": "PASS",
        "gpu_model": device_name,
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "total_vram_gb": round(vram_gb, 2),
        "sm_80_plus": is_ampere_or_newer,
        "torch_bf16_supported": bf16_supported,
        "gemm_throughput_tflops": round(tflops, 2),
        "gemm_latency_ms": round(t_gemm * 1000, 2),
        "gemm_output_dtype": str(C.dtype),
        "forward_backward_loss": round(loss.item(), 4),
        "weight_gradient_dtype": str(layer[0].weight.grad.dtype),
        "gradient_has_nan": bool(torch.isnan(layer[0].weight.grad).any().item()),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "smoke_test_runtime_seconds": round(runtime_sec, 2),
        "hf_hardware": "a100-large",
        "hourly_rate_usd": 2.50
    }
    
    print("\\n=== [SMOKE TEST PASSED] Hardware is certified for Stage B BF16 execution ===", flush=True)
    print(f"Peak VRAM: {peak_vram_gb:.4f} GB | Test Runtime: {runtime_sec:.2f}s", flush=True)
    print("RESULTS_JSON_START", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("RESULTS_JSON_END", flush=True)

with open("smoke_test_receipt.json", "w") as f:
    json.dump(results, f, indent=2)

def get_status():
    return json.dumps(results, indent=2)

demo = gr.Interface(
    fn=get_status,
    inputs=[],
    outputs=gr.JSON(),
    title="Polyglot-MANSU A100 BF16 Smoke Test Results"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
"""

README_CONTENT = """---
title: Polyglot-MANSU A100 BF16 Smoke Test
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.1
python_version: '3.12'
app_file: app.py
pinned: false
---

Internal automated hardware smoke test for NVIDIA A100 80GB Native BF16 support.
"""

def main():
    api = HfApi()
    print("==================================================================")
    print(f"=== Dispatching A100 BF16 Smoke Test to Space: {REPO_ID} ===")
    print("==================================================================")
    
    # Verify space exists and check requested hardware
    runtime = api.get_space_runtime(REPO_ID)
    print(f"Current Space Stage: {runtime.stage} | Requested Hardware: {runtime.requested_hardware}")
    
    # 1. Upload files
    print("\nUploading app.py, requirements.txt, and README.md to Hugging Face...")
    api.upload_file(
        path_or_fileobj=APP_PY_CONTENT.encode("utf-8"),
        path_in_repo="app.py",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj=REQUIREMENTS_CONTENT.encode("utf-8"),
        path_in_repo="requirements.txt",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj=README_CONTENT.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space"
    )
    print("Upload complete. Restarting space...")
    try:
        api.restart_space(REPO_ID)
    except Exception as e:
        print(f"Restart notice: {e}")
    print("Upload complete. Space rebuild triggered.")
    
    # Set sleep time to 300s (5m) as automated guardrail against orphaned GPU runs
    try:
        api.set_space_sleep_time(REPO_ID, sleep_time=300)
    except Exception as e:
        print(f"Note on sleep time: {e}")

    # 2. Monitor space stage & stream logs
    t_dispatch = time.time()
    max_wait_seconds = 600  # 10 minutes max
    poll_interval = 5
    smoke_results = None
    full_logs = ""
    
    print("\nMonitoring A100 provisioning and execution...")
    while time.time() - t_dispatch < max_wait_seconds:
        runtime = api.get_space_runtime(REPO_ID)
        stage = runtime.stage
        elapsed = time.time() - t_dispatch
        print(f"[{elapsed:03.0f}s] Stage: {stage} | Hardware: {runtime.hardware} (requested: {runtime.requested_hardware})")
        
        # Check logs if building or running
        if stage in ["BUILDING", "RUNNING", "APP_STARTING"]:
            try:
                log_chunks = list(api.fetch_space_logs(REPO_ID))
                full_logs = "".join(log_chunks)
                if "RESULTS_JSON_START" in full_logs and "RESULTS_JSON_END" in full_logs:
                    start_idx = full_logs.find("RESULTS_JSON_START") + len("RESULTS_JSON_START")
                    end_idx = full_logs.find("RESULTS_JSON_END")
                    raw_json = full_logs[start_idx:end_idx].strip()
                    smoke_results = json.loads(raw_json)
                    print("\n>>> Detected completed smoke test results in logs! <<<")
                    break
            except Exception as e:
                # Log reading can be transiently empty during container spin-up
                pass
                
        if stage in ["BUILD_ERROR", "RUNTIME_ERROR"]:
            print(f"FATAL: Space entered failure stage: {stage}")
            try:
                print("--- SPACE LOGS ---")
                print("".join(list(api.fetch_space_logs(REPO_ID))))
            except Exception:
                pass
            break
            
        time.sleep(poll_interval)
        
    total_elapsed_seconds = round(time.time() - t_dispatch, 2)
    
    # 3. IMMEDIATELY Pause Space to Stop GPU Billing
    print("\n==================================================================")
    print("=== IMMEDIATE ACTION: Pausing A100 GPU Space to Stop Billing ===")
    print("==================================================================")
    try:
        api.pause_space(REPO_ID)
        print("Space PAUSED successfully. GPU billing stopped.")
    except Exception as e:
        print(f"Warning on pause: {e}. Attempting hardware request downgrade to cpu-basic...")
        try:
            api.request_space_hardware(REPO_ID, hardware="cpu-basic")
            print("Space hardware downgraded to cpu-basic.")
        except Exception as e2:
            print(f"Warning on downgrade: {e2}")

    # 4. Save and report results
    if smoke_results:
        # Calculate billing cost
        # A100-large is $2.50/hr ($0.041667/min)
        total_minutes = total_elapsed_seconds / 60.0
        billing_cost_usd = total_minutes * 0.041667
        smoke_results["total_session_runtime_seconds"] = total_elapsed_seconds
        smoke_results["total_session_cost_usd"] = round(billing_cost_usd, 5)
        
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = results_dir / "smoke_test_a100_receipt.json"
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(smoke_results, f, indent=2)
            
        logs_path = results_dir / "smoke_test_a100.log"
        with open(logs_path, "w", encoding="utf-8") as f:
            f.write(full_logs)
            
        print("\n==================================================================")
        print("=== A100 BF16 SMOKE TEST CERTIFICATION REPORT ===")
        print("==================================================================")
        print(f"GPU Model:              {smoke_results.get('gpu_model')}")
        print(f"Compute Capability:     {smoke_results.get('compute_capability')} (sm_80+ certified: {smoke_results.get('sm_80_plus')})")
        print(f"Total VRAM:             {smoke_results.get('total_vram_gb')} GB")
        print(f"BF16 Supported:         {smoke_results.get('torch_bf16_supported')}")
        print(f"GEMM Throughput:        {smoke_results.get('gemm_throughput_tflops')} TFLOPS ({smoke_results.get('gemm_latency_ms')} ms)")
        print(f"GEMM Output Dtype:      {smoke_results.get('gemm_output_dtype')}")
        print(f"Forward-Backward Loss:  {smoke_results.get('forward_backward_loss')}")
        print(f"Weight Gradient Dtype:  {smoke_results.get('weight_gradient_dtype')}")
        print(f"Gradient NaN Detected:  {smoke_results.get('gradient_has_nan')}")
        print(f"Peak GPU VRAM:          {smoke_results.get('peak_vram_gb')} GB")
        print(f"Test Execution Time:    {smoke_results.get('smoke_test_runtime_seconds')}s")
        print(f"Total Session Duration: {smoke_results.get('total_session_runtime_seconds')}s ({total_minutes:.2f} min)")
        print(f"Billed Expenditure:     ${smoke_results.get('total_session_cost_usd')} USD")
        print(f"Receipt Saved:          {receipt_path}")
        print(f"Logs Saved:             {logs_path}")
        print("==================================================================")
    else:
        print("FAIL: Smoke test results could not be retrieved from space logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
