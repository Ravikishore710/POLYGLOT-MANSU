"""
scripts/compile_verified_tables.py
Compiles verified Markdown benchmark tables and publication figures strictly from
executed evaluation JSON artifacts.
Enforces Law 2: Reads only hashed JSON files; zero manual numbers permitted.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

def hash_file(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compile_main_results_table(json_paths: Dict[str, Path]) -> str:
    """
    Builds the main 9-checkpoint comparison table across all 6 pillars.
    """
    header = (
        "| Model Checkpoint | Method Type | SafeTune Badge | Forget Acc (BF16) | Forget Acc (NF4) | "
        "Delta_PTQ | Hinglish/Telugu ASR | CAD | Global-MMLU | Artifact SHA-256 |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for model_name, path in json_paths.items():
        if not path.exists():
            continue
        sha = hash_file(path)[:8]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        badge = data.get("safetune_audit_badge", "N/A")
        f_bf16 = data.get("pillar_1_direct_forget_acc", 0.0)
        f_nf4 = f_bf16 + data.get("pillar_4_delta_ptq_nf4", 0.0)
        d_ptq = data.get("pillar_4_delta_ptq_nf4", 0.0)
        asr_cm = data.get("pillar_2_polyglot_asr_mean", 0.0)
        cad = data.get("pillar_5_cad", 0.0)
        mmlu = data.get("pillar_6_mmlu_utility", 0.0)

        row = (
            f"| `{model_name}` | Baseline/Ours | {badge} | {f_bf16:.2%} | {f_nf4:.2%} | "
            f"{d_ptq:+.2%} | {asr_cm:.1%} | {cad:.2f} | {mmlu:.1%} | `{sha}` |"
        )
        rows.append(row)

    return header + "\n".join(rows) + "\n"

def main():
    results_dir = Path("results/audit")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all summary files
    summaries = {}
    
    # Check baseline receipts
    baseline_receipt = Path("audit_for_carbon/09_stage3_baseline_evaluation_receipt.json")
    if baseline_receipt.exists():
        summaries["tiny_aya_fire_baseline_unmodified"] = baseline_receipt

    for p in results_dir.glob("*_audit_summary.json"):
        model_key = p.stem.replace("_audit_summary", "")
        summaries[model_key] = p

    # If results/audit is empty, check audit_for_carbon
    carbon_dir = Path("audit_for_carbon")
    for p in carbon_dir.glob("*proof.json"):
        summaries[p.stem] = p

    md_table = compile_main_results_table(summaries)
    
    out_file = Path("docs/VERIFIED_BENCHMARK_TABLES.md")
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Verified Benchmark Tables\n\nGenerated deterministically from raw JSON artifacts.\n\n")
        f.write(md_table)

    print(f"Compiled verified tables to {out_file} from {len(summaries)} JSON artifacts.")

if __name__ == "__main__":
    main()
