import urllib.request
from huggingface_hub import HfApi

REPO_ID = "Ravikishore710/polyglot-mansu-fire-weights-private"
api = HfApi()

print(f"Updating settings of {REPO_ID} to Public (private=False)...")
api.update_repo_settings(repo_id=REPO_ID, private=False, repo_type="model")

# Verify repo info
info = api.repo_info(repo_id=REPO_ID, repo_type="model")
print(f"Updated Repo: {info.id} | Private: {info.private}")

# Verify anonymous public HTTP access
urls_to_test = [
    f"https://huggingface.co/{REPO_ID}",
    f"https://huggingface.co/{REPO_ID}/raw/main/config.json",
    f"https://huggingface.co/{REPO_ID}/raw/main/checksums.json",
    f"https://huggingface.co/{REPO_ID}/resolve/main/model.safetensors"
]

print("\nVerifying anonymous public access without auth headers...")
for url in urls_to_test:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req)
        print(f"  [HTTP {resp.status}] PUBLIC ACCESSIBLE: {url}")
    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
