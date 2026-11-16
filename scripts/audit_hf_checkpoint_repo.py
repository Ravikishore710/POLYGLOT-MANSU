import os
import re
import json
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Ravikishore710/polyglot-mansu-fire-weights-private"
api = HfApi()

print(f"=== Inspecting {REPO_ID} ===")
info = api.repo_info(repo_id=REPO_ID, repo_type="model")
print(f"Repo ID: {info.id} | Private: {info.private}")

files_info = api.list_repo_tree(repo_id=REPO_ID, repo_type="model")
file_names = []
for item in files_info:
    size_str = f"{item.size:,} bytes" if hasattr(item, 'size') and item.size is not None else "unknown size"
    lfs_str = " (LFS)" if hasattr(item, 'lfs') and item.lfs else ""
    print(f"  - {item.path}: {size_str}{lfs_str}")
    file_names.append(item.path)

# Scan text files for any secrets
secret_patterns = [
    r"hf_[A-Za-z0-9]{34,}",
    r"sk-[A-Za-z0-9]{32,}",
    r"AIza[0-9A-Za-z-_]{35}",
    r"api[_-]?key",
    r"bearer\s+",
    r"ghp_[A-Za-z0-9]{36}"
]

print("\n=== Scanning text files for secrets / credentials ===")
text_files = [f for f in file_names if f.endswith(('.json', '.jinja', '.yaml', '.txt', '.md', '.py')) and 'sample_log' not in f]
# Also scan sample log top/bottom
for tf in text_files + ['training_sample_log_tiny_aya_fire_mansu.json']:
    try:
        local_f = hf_hub_download(repo_id=REPO_ID, filename=tf, repo_type="model")
        with open(local_f, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        found_secrets = []
        for pat in secret_patterns:
            matches = re.findall(pat, content, re.IGNORECASE)
            # filter false positives like "api_key": null or descriptions
            for m in matches:
                if len(m) > 10 and not any(ign in m.lower() for ign in ['api_key": null', 'api_key: null', 'gemini_api_key']):
                    found_secrets.append(m)
        
        if found_secrets:
            print(f"  [ALERT] Potential secret in {tf}: {found_secrets}")
        else:
            print(f"  [CLEAN] {tf} verified free of credentials/tokens.")
    except Exception as e:
        print(f"  [ERROR] {tf}: {e}")
