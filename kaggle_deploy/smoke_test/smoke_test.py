"""
Polyglot-MANSU: Stage 1 Inference Smoke Test
Target: CohereLabs/tiny-aya-fire (3.35B)
Verifies: HF Secret Auth -> Gated Repo Access -> GPU Memory -> Real Token Generation
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
import torch

print("=" * 70)
print("=== [POLYGLOT-MANSU] STAGE 1: GPU INFERENCE SMOKE TEST ===")
print("=" * 70)

log_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "stage": "smoke_test",
    "model_id": "CohereLabs/tiny-aya-fire"
}

# -------------------------------------------------------------------------
# STEP 1: Verify Hardware & GPU Environment
# -------------------------------------------------------------------------
print("\n[Step 1/5] Checking GPU Accelerator Environment...")
if not torch.cuda.is_available():
    print("❌ ERROR: No GPU detected. Please go to Notebook Options -> Accelerator -> Select 'GPU T4 x2'.")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
device_count = torch.cuda.device_count()
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"  ✓ GPU Detected: {device_name} (Total Devices: {device_count})")
print(f"  ✓ Device 0 Total VRAM: {total_vram_gb:.2f} GB")

log_data["gpu"] = {
    "device_name": device_name,
    "device_count": device_count,
    "total_vram_gb": round(total_vram_gb, 2)
}

# -------------------------------------------------------------------------
# STEP 2: Configure HF Authentication
# -------------------------------------------------------------------------
print("\n[Step 2/5] Configuring Hugging Face Authentication...")
hf_token = os.environ.get("HF_TOKEN", "")

# Fallback: check Kaggle User Secrets if running in Kaggle notebook
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
    except Exception:
        pass

if hf_token:
    print("  ✓ HF Token retrieved from secure environment/secrets (Token value is kept hidden).")
else:
    print("  ⚠ Notice: No HF_TOKEN environment variable or Kaggle secret detected.")
    print("    Proceeding with local Hugging Face cache authentication if available.")

# -------------------------------------------------------------------------
# STEP 3: Verify Hugging Face Account & Gated License Access
# -------------------------------------------------------------------------
print("\n[Step 3/5] Verifying Hugging Face Account & Gated Model Access...")
try:
    from huggingface_hub import HfApi, login
    import huggingface_hub.errors as hf_errors

    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        api = HfApi(token=hf_token)
    else:
        api = HfApi()
    user_info = api.whoami()
    username = user_info.get("name") or user_info.get("preferred_username") or "Authenticated User"
    print(f"  ✓ Authenticated as Hugging Face user: {username}")
    log_data["hf_user"] = username

    print("  -> Checking access permission for CohereLabs/tiny-aya-fire...")
    model_info = api.model_info("CohereLabs/tiny-aya-fire")
    print("  ✓ Access confirmed: You have accepted Cohere's terms for this model.")
    log_data["gated_access_status"] = "GRANTED"

except hf_errors.GatedRepoError:
    print("❌ GATED ACCESS ERROR: You have not accepted the model access conditions yet.")
    sys.exit(1)
except hf_errors.HfHubHTTPError as e:
    print(f"❌ AUTHENTICATION ERROR: Hugging Face rejected token. Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ UNEXPECTED HUB ERROR: {e}")
    sys.exit(1)

# -------------------------------------------------------------------------
# STEP 4: Load Tokenizer & Model Weights into GPU Memory
# -------------------------------------------------------------------------
print("\n[Step 4/5] Loading Tokenizer & Model Weights into GPU (BF16/FP16)...")
MODEL_ID = "CohereLabs/tiny-aya-fire"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("  -> Downloading/Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"  -> Loading Model weights in {dtype} with device_map='auto'...")
    start_load_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto"
    )
    load_duration = time.time() - start_load_time
    print(f"  ✓ Model successfully loaded onto GPU in {load_duration:.1f} seconds.")

    allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved_vram = torch.cuda.memory_reserved(0) / (1024 ** 3)
    print(f"  ✓ GPU VRAM Usage after load: {allocated_vram:.2f} GB allocated ({reserved_vram:.2f} GB reserved)")

    log_data["model_load"] = {
        "status": "SUCCESS",
        "duration_sec": round(load_duration, 1),
        "allocated_vram_gb": round(allocated_vram, 2),
        "dtype": str(dtype)
    }

except torch.cuda.OutOfMemoryError:
    print("❌ CUDA OUT OF MEMORY: Model weights exceeded available GPU VRAM.")
    sys.exit(1)
except Exception as e:
    print(f"❌ MODEL LOADING ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# -------------------------------------------------------------------------
# STEP 5: Run Single Test Prompt Generation
# -------------------------------------------------------------------------
print("\n[Step 5/5] Running Live Test Inference...")
test_prompt = "Hello, introduce yourself briefly in one sentence."
print(f"  -> Prompt: \"{test_prompt}\"")

try:
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda:0")
    input_ids = inputs["input_ids"]
    print(f"  -> Input token count: {input_ids.shape[1]}")

    start_gen_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_duration = time.time() - start_gen_time

    # Decode only the newly generated response tokens
    new_tokens = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not generated_text:
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    print("\n" + "-" * 70)
    print(f"✨ GENERATED RESPONSE:\n{generated_text}")
    print("-" * 70)
    print(f"  ✓ Generation finished in {gen_duration:.2f} seconds.")

    log_data["inference"] = {
        "status": "SUCCESS",
        "prompt": test_prompt,
        "response": generated_text,
        "generation_time_sec": round(gen_duration, 2)
    }

except Exception as e:
    print(f"❌ INFERENCE GENERATION ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

out_path = Path("/kaggle/working/smoke_test_receipt.json")
try:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n✓ Execution receipt saved to: {out_path}")
except Exception:
    pass

print("\n" + "=" * 70)
print("🎉 STAGE 1 COMPLETE: CohereLabs/tiny-aya-fire IS FULLY OPERATIONAL ON GPU!")
print("=" * 70)
