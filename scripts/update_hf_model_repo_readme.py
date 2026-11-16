from pathlib import Path
from huggingface_hub import HfApi

YAML_HEADER = """---
language:
  - en
  - hi
  - te
  - bn
  - ta
  - ar
  - es
  - fr
license: cc-by-nc-4.0
tags:
  - unlearning
  - safety
  - alignment
  - tiny-aya
  - cohere
  - lexsi-hackathon
  - circuit-attribution
  - quantization
base_model: CohereLabs/tiny-aya-fire
pipeline_tag: text-generation
---

<p align="center">
  <img src="https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/resolve/main/polyglot-mansu-cover.png" alt="Polyglot-MANSU Cover" width="100%"/>
</p>

"""

article_content = Path("docs/POLYGLOT_MANSU_HF_ARTICLE.md").read_text(encoding="utf-8")
full_readme = YAML_HEADER + article_content

temp_readme_path = Path("temp_hf_model_readme.md")
temp_readme_path.write_text(full_readme, encoding="utf-8")

api = HfApi()
repo_id = "Ravikishore710/polyglot-mansu-tiny-aya-fire"

print(f"Uploading updated README.md to {repo_id}...")
api.upload_file(
    path_or_fileobj=str(temp_readme_path),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Update model card README with authentic A100 Stage B.2 checkpoint disclosures and compute accounting"
)
print("Successfully uploaded README.md to Hugging Face model repository!")

# Also upload Stage B.2 receipts to this repo so judges have them directly in the repo
stage_b2_receipts = [
    ("results/audit/hardened_fire/execution_receipt_tiny_aya_fire_mansu_export.json", "execution_receipt_tiny_aya_fire_mansu_export.json"),
    ("results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json", "training_sample_log_tiny_aya_fire_mansu.json"),
    ("results/smoke_test_a100_receipt.json", "smoke_test_a100_receipt.json")
]

for local_path, repo_path in stage_b2_receipts:
    p = Path(local_path)
    if p.exists():
        print(f"Uploading {local_path} as {repo_path}...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add Stage B.2 execution telemetry: {repo_path}"
        )

# Clean up temporary file
temp_readme_path.unlink()
print("All model repo assets updated successfully!")
