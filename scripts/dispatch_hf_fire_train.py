#!/usr/bin/env python3
"""
scripts/dispatch_hf_fire_train.py
Autonomous Hugging Face A100 Stage B.2 Fire Training & Checkpoint Dispatcher.
1. Uploads stage_b_space_bundle to private space Ravikishore710/polyglot-mansu-smoke-test.
2. Restarts space on Nvidia A100-80GB.
3. Streams logs in real time, tracking all 300 steps under Option B.
4. Detects STAGE_B2_EXPORT_COMPLETE and extracts execution receipt.
5. IMMEDIATELY pauses space to eliminate idle GPU billing.
6. Downloads exported checkpoint to models/tiny_aya_fire_mansu_weights/.
7. Downloads execution receipt & sample log to results/audit/hardened_fire/.
8. Re-computes SHA-256 checksums locally and verifies bit-for-bit authenticity.
9. Prints comprehensive report and stops for user review.
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

REPO_ID = "Ravikishore710/polyglot-mansu-smoke-test"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    api = HfApi()
    print("==================================================================")
    print(f"=== Dispatching Stage B.2 Fire Training to Space: {REPO_ID} ===")
    print("==================================================================")

    bundle_path = Path("stage_b_space_bundle")
    assert bundle_path.exists(), "FATAL: stage_b_space_bundle does not exist!"

    # 1. Upload stage_b_space_bundle
    print("\nUploading stage_b_space_bundle to Hugging Face Space...")
    api.upload_folder(
        folder_path=str(bundle_path),
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy Stage B.2 Fire unlearning and export pipeline"
    )
    print("Upload complete. Restarting space on A100...")

    # Restart space
    try:
        api.restart_space(REPO_ID)
    except Exception as e:
        print(f"Restart notice: {e}")

    # Set safety sleep time (10 min)
    try:
        api.set_space_sleep_time(REPO_ID, sleep_time=600)
    except Exception as e:
        print(f"Sleep time notice: {e}")

    # 2. Monitor execution and stream logs
    t_dispatch = time.time()
    max_wait_seconds = 1200  # 20 minutes max
    poll_interval = 5
    fire_receipt = None
    full_logs = ""
    last_log_len = 0

    print("\nMonitoring A100 execution and streaming training logs...")
    while time.time() - t_dispatch < max_wait_seconds:
        runtime = api.get_space_runtime(REPO_ID)
        stage = runtime.stage
        elapsed = time.time() - t_dispatch

        # Stream logs if building or running
        if stage in ["BUILDING", "RUNNING", "APP_STARTING"]:
            try:
                log_chunks = list(api.fetch_space_logs(REPO_ID))
                full_logs = "".join(log_chunks)
                if len(full_logs) > last_log_len:
                    new_text = full_logs[last_log_len:]
                    # Print new training progress lines
                    for line in new_text.splitlines():
                        if any(k in line for k in ["Epoch", "Step", "Hardware", "Checkpoint", "Reload", "PASSED", "C*", "STAGE_B2"]):
                            print(f"  [REMOTE A100] {line}", flush=True)
                    last_log_len = len(full_logs)

                if "RECEIPT_JSON_START" in full_logs and "RECEIPT_JSON_END" in full_logs:
                    start_idx = full_logs.find("RECEIPT_JSON_START") + len("RECEIPT_JSON_START")
                    end_idx = full_logs.find("RECEIPT_JSON_END")
                    raw_json = full_logs[start_idx:end_idx].strip()
                    fire_receipt = json.loads(raw_json)
                    print("\n>>> Detected STAGE_B2_EXPORT_COMPLETE and receipt in logs! <<<", flush=True)
                    break
            except Exception as e:
                pass

        if stage in ["BUILD_ERROR", "RUNTIME_ERROR"]:
            print(f"FATAL: Space entered failure stage: {stage}", flush=True)
            try:
                print("--- SPACE LOGS ---")
                print("".join(list(api.fetch_space_logs(REPO_ID))))
            except Exception:
                pass
            break

        time.sleep(poll_interval)

    total_session_seconds = round(time.time() - t_dispatch, 2)

    # 3. IMMEDIATELY Pause Space to Stop GPU Billing
    print("\n==================================================================")
    print("=== IMMEDIATE ACTION: Pausing A100 GPU Space to Stop Billing ===")
    print("==================================================================")
    try:
        api.pause_space(REPO_ID)
        print("Space PAUSED successfully. GPU billing stopped.")
    except Exception as e:
        print(f"Warning on pause: {e}")

    # 4. Download Exported Checkpoint and Receipts
    if fire_receipt:
        print("\n==================================================================")
        print("=== Downloading Exported Checkpoint & Receipts to Local Repo ===")
        print("==================================================================")

        local_weights_dir = Path("models/tiny_aya_fire_mansu_weights")
        local_weights_dir.mkdir(parents=True, exist_ok=True)
        local_results_dir = Path("results/audit/hardened_fire")
        local_results_dir.mkdir(parents=True, exist_ok=True)

        TARGET_MODEL_REPO = "Ravikishore710/polyglot-mansu-fire-weights-private"
        # Download checkpoint files from private model repo
        print(f"Pulling checkpoint files from {TARGET_MODEL_REPO} into {local_weights_dir}...")
        snapshot_download(
            repo_id=TARGET_MODEL_REPO,
            repo_type="model",
            local_dir=str(local_weights_dir)
        )
        for f in local_weights_dir.glob("*"):
            if f.is_file():
                print(f"  Downloaded: {f.name} ({f.stat().st_size / (1024**2):.2f} MB)")

        # Download receipt and sample log
        for filename in ["execution_receipt_tiny_aya_fire_mansu_export.json", "training_sample_log_tiny_aya_fire_mansu.json"]:
            try:
                remote_f = hf_hub_download(repo_id=TARGET_MODEL_REPO, filename=filename, repo_type="model")
                dest = local_results_dir / filename
                dest.write_bytes(Path(remote_f).read_bytes())
                print(f"  Downloaded: {filename}")
            except Exception as e:
                print(f"  Notice downloading {filename}: {e}")

        # Save training logs to results
        with open(local_results_dir / "stage_b2_fire_training.log", "w", encoding="utf-8") as f:
            f.write(full_logs)

        # 5. Local SHA-256 Checksum Verification
        print("\nVerifying SHA-256 checksums locally...")
        local_manifest = {}
        manifest_path = local_weights_dir / "checksums.json"
        remote_checksums = fire_receipt["checkpoint_export"]["checksums"]
        mismatch_count = 0

        for f in sorted(local_weights_dir.glob("*")):
            if f.is_file() and f.name != "checksums.json":
                h = compute_sha256(f)
                local_manifest[f.name] = h
                expected = remote_checksums.get(f.name)
                match = (h == expected)
                status = "BIT-EXACT" if match else "MISMATCH"
                if not match:
                    mismatch_count += 1
                print(f"  {f.name}: {h[:16]}... [{status}]")

        assert mismatch_count == 0, f"FATAL: {mismatch_count} checksum mismatches detected!"
        print(f"Local verification PASSED: All {len(local_manifest)} files match remote export bit-for-bit.")

        # Compute billing details
        total_session_minutes = total_session_seconds / 60.0
        session_cost_usd = total_session_minutes * 0.041667
        fire_receipt["total_session_runtime_seconds"] = total_session_seconds
        fire_receipt["total_session_cost_usd"] = round(session_cost_usd, 5)

        with open(local_results_dir / "execution_receipt_tiny_aya_fire_mansu_export.json", "w", encoding="utf-8") as f:
            json.dump(fire_receipt, f, indent=2)

        print("\n==================================================================")
        print("=== STAGE B.2 FIRE UNLEARNING & CHECKPOINT EXPORT REPORT ===")
        print("==================================================================")
        print(f"Model ID:                {fire_receipt.get('model_id')}")
        print(f"Hardware Used:           {fire_receipt.get('hardware')}")
        print(f"Compute Capability:      {fire_receipt.get('compute_capability')}")
        print(f"Sampling Methodology:   {fire_receipt.get('sampling_methodology')}")
        print(f"Steps Trained:           {fire_receipt.get('steps_trained')} (Epochs: {fire_receipt.get('epochs')})")
        print(f"Forget Coverage:         {fire_receipt.get('unique_forget_samples_seen')}/{fire_receipt.get('total_forget_corpus_size')} unique samples")
        print(f"Per-Epoch Retain Starts: {fire_receipt.get('per_epoch_retain_initial_ids')}")
        print(f"Reload Verification:     {fire_receipt['checkpoint_export'].get('reload_verification')} (max diff: {fire_receipt['checkpoint_export'].get('max_parameter_difference')})")
        print(f"Peak VRAM:               {fire_receipt.get('peak_vram_gb')} GB")
        print(f"Training Runtime:        {fire_receipt.get('runtime_seconds')}s")
        print(f"Total Session Duration:  {total_session_seconds}s ({total_session_minutes:.2f} min)")
        print(f"Billed Session Cost:     ${session_cost_usd:.5f} USD")
        print(f"Checkpoint Location:     {local_weights_dir.resolve()}")
        print(f"File Inventory:          {list(local_manifest.keys())}")
        print(f"Receipt Path:            {local_results_dir / 'execution_receipt_tiny_aya_fire_mansu_export.json'}")
        print("==================================================================")

    else:
        print("FATAL: Stage B.2 execution did not produce completion signature.")
        sys.exit(1)

if __name__ == "__main__":
    main()
