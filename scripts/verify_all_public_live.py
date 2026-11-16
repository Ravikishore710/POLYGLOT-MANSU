import urllib.request
import urllib.error

urls = {
    "1. GitHub Source Code": "https://github.com/rahulratan12/Polyglot-MANSU",
    "2. HF Models Collection": "https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1",
    "3. HF Datasets Collection": "https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5",
    "4. Primary Model Checkpoint (6.70 GB)": "https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private",
    "5. Model Checkpoint Direct Safetensors": "https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private/resolve/main/model.safetensors",
    "6. Model Telemetry & Audit Repo": "https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire",
    "7. Curated Multilingual Dataset": "https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated",
    "8. Interactive Demo Arena (Space)": "https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena",
    "9. Published Research Article (Discussion #2)": "https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/discussions/2"
}

print("=== Anonymous Public Accessibility Audit (Zero Auth Headers) ===")
all_pass = True
for name, url in urls.items():
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        resp = urllib.request.urlopen(req)
        print(f"[HTTP {resp.status} OK] PUBLIC: {name} -> {url}")
    except urllib.error.HTTPError as e:
        print(f"[HTTP {e.code} FAIL] INACCESSIBLE: {name} -> {url}")
        all_pass = False
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        all_pass = False

print("\nFinal Verdict:", "ALL ASSETS 100% PUBLICLY ACCESSIBLE!" if all_pass else "SOME ASSETS INACCESSIBLE")
