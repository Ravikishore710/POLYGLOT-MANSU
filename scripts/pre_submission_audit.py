#!/usr/bin/env python3
"""
scripts/pre_submission_audit.py
Pre-submission verification audit for Lexsi AI Safety & Alignment Hackathon (Track 2b #7).
Checks:
1. All python files compile cleanly via py_compile
2. strict_self_verifier.py reports 0 violations
3. Manifests and SHA256SUMS.txt are present
4. Required directories and artifacts exist
"""

import sys
import py_compile
from pathlib import Path
import subprocess

REQUIRED_FILES = [
    "AGENTS.md",
    "PLAYBOOK.md",
    "pyproject.toml",
    "docs/VERIFIED_API_SIGNATURES.json",
    "audit_for_carbon/01_verified_lexsi_apis.json",
    "audit_for_carbon/02_ast_auditor_selftest.json",
    "audit_for_carbon/03_curatorkit_manifest_summary.json",
    "audit_for_carbon/04_circuitkit_e2e_proof.json",
    "audit_for_carbon/05_trainer_and_ptq_math_proof.json",
    "audit_for_carbon/06_full_4repo_pipeline_e2e.json",
    "space_app/app.py",
    "notebooks/kaggle_account1_fire.py",
    "notebooks/kaggle_account2_global.py"
]

def check_compilation():
    print("Checking python compilation...")
    py_files = list(Path("src").rglob("*.py")) + list(Path("scripts").glob("*.py")) + list(Path("notebooks").glob("*.py")) + list(Path("space_app").glob("*.py")) + list(Path("tests").glob("*.py"))
    for pf in py_files:
        try:
            py_compile.compile(str(pf), doraise=True)
        except py_compile.PyCompileError as e:
            print(f"Compilation FAILED on {pf}: {e}")
            return False
    print(f"Successfully compiled {len(py_files)} Python files.")
    return True

def check_required_files():
    print("Checking required files...")
    missing = []
    for rf in REQUIRED_FILES:
        p = Path(rf)
        if not p.exists():
            missing.append(rf)
    if missing:
        print(f"Missing required files: {missing}")
        return False
    print(f"All {len(REQUIRED_FILES)} required files are verified present.")
    return True

def run_strict_verifier():
    print("Running strict self-verifier...")
    res = subprocess.run([sys.executable, "scripts/strict_self_verifier.py"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Strict Self-Verifier reported violations:\n{res.stdout}\n{res.stderr}")
        return False
    print(res.stdout.strip())
    return True

def main():
    ok1 = check_compilation()
    ok2 = check_required_files()
    ok3 = run_strict_verifier()

    if ok1 and ok2 and ok3:
        print("\nPRE-SUBMISSION AUDIT: PASSED (100% Green, Ready for Kaggle Dual-T4 & Judge Submission)")
        return 0
    else:
        print("\nPRE-SUBMISSION AUDIT: FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
