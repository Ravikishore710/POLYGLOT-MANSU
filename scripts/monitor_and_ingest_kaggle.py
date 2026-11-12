#!/usr/bin/env python3
"""
scripts/monitor_and_ingest_kaggle.py
Automated monitoring and artifact ingestion script for Kaggle training runs.
"""

import time
import json
import subprocess
from pathlib import Path
import kaggle

def get_status_attr(obj, attr: str):
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)

def check_kernels(api):
    kernels = [
        {"id": "rahulratan12/polyglot-mansu-fire-run", "target": "fire", "out_dir": Path("results/kaggle_fire")},
        {"id": "rahulratan12/polyglot-mansu-global-run", "target": "global", "out_dir": Path("results/kaggle_global")}
    ]
    results = {}
    for k in kernels:
        res = api.kernels_status(k["id"])
        st = str(get_status_attr(res, "status"))
        fail = str(get_status_attr(res, "failure_message") or get_status_attr(res, "failureMessage"))
        results[k["id"]] = {"status": st, "failure": fail, "target": k["target"], "out_dir": k["out_dir"]}
        print(f"[{time.strftime('%X')}] Kernel: {k['id']} | Status: {st} | Failure: {fail}")
    return results

def main():
    api = kaggle.KaggleApi()
    api.authenticate()
    print("=== [MONITOR] Checking status of Kaggle GPU kernels ===")
    status_map = check_kernels(api)

if __name__ == "__main__":
    main()
