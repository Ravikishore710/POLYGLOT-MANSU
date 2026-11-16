import subprocess
import re

res = subprocess.run(["git", "diff", "origin/main...main"], capture_output=True, text=True, encoding="utf-8")
diff_text = res.stdout

secret_patterns = [
    r"hf_[A-Za-z0-9]{34,}",
    r"sk-[A-Za-z0-9]{32,}",
    r"AIza[0-9A-Za-z-_]{35}",
    r"ghp_[A-Za-z0-9]{36}",
    r"(?:api_key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9_-]{16,}['\"]"
]

print("=== Scanning git diff origin/main...main for secrets ===")
found_secrets = []
for pat in secret_patterns:
    for m in re.finditer(pat, diff_text, re.IGNORECASE):
        match_str = m.group(0)
        # Check if line is an addition
        found_secrets.append(match_str)

if found_secrets:
    print(f"ALERT: Potential secrets found: {found_secrets}")
else:
    print("CLEAN: Zero secrets or tokens found in git diff origin/main...main!")
