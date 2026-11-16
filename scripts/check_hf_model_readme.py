from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
repo_id = "Ravikishore710/polyglot-mansu-tiny-aya-fire"

try:
    readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    print("README.md length:", len(content))
    print("\nFirst 500 chars:\n", content[:500])
    print("\nLast 500 chars:\n", content[-500:])
except Exception as e:
    print(f"Error fetching README: {e}")
