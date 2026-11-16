import os
import re
from pathlib import Path

toolkits = ["curatorkit", "circuitkit", "safetune", "auditkit"]

src_dir = Path("src")
py_files = list(src_dir.rglob("*.py"))

print("=== Scanning src/ for Toolkit Usages ===")
for tk in toolkits:
    print(f"\n--- Toolkit: {tk} ---")
    matches = []
    for f in py_files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        found = re.findall(rf"(?:from\s+{tk}[^\n]+|import\s+{tk}[^\n]*)", content)
        if found:
            print(f"File: {f}")
            for m in found:
                print(f"  {m}")
