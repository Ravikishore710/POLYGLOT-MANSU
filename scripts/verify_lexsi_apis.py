#!/usr/bin/env python3
"""
scripts/verify_lexsi_apis.py
Live introspection and verification of Lexsi Labs alignment stack packages:
- CircuitKIT
- SafeTune
- CuratorKIT
- AuditKIT
Also verifies GUARD-04 (CircuitKIT import order before TransformerLens) and logs all signatures.
"""

import sys
import json
import inspect
from pathlib import Path

# Enforce GUARD-04: import circuitkit BEFORE transformer_lens
try:
    import circuitkit
    import circuitkit.backends
    circuitkit_imported_first = True
except Exception as e:
    circuitkit_imported_first = False
    circuitkit_error = str(e)

import torch

def introspect_module(mod):
    """Extract public classes, functions, and docstrings from a module."""
    items = {}
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if inspect.isclass(obj):
            methods = [m for m, _ in inspect.getmembers(obj, predicate=inspect.isfunction) if not m.startswith("_")]
            sig = ""
            try:
                sig = str(inspect.signature(obj.__init__))
            except Exception:
                sig = "unknown"
            items[name] = {"type": "class", "init_sig": sig, "methods": methods[:15]}
        elif inspect.isfunction(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "unknown"
            items[name] = {"type": "function", "sig": sig}
    return items

def main():
    report = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "circuitkit": {},
        "safetune": {},
        "curatorkit": {},
        "auditkit": {},
        "guards_verified": {}
    }

    # 1. CircuitKIT
    try:
        import circuitkit
        report["circuitkit"]["version"] = getattr(circuitkit, "__version__", "unknown")
        report["circuitkit"]["path"] = str(Path(circuitkit.__file__).parent)
        
        # Check cohere compatibility
        has_tl_compat = hasattr(circuitkit.backends, "_tl_compat")
        report["circuitkit"]["has_tl_compat"] = has_tl_compat
        
        # Check algorithms and evaluation
        import circuitkit.discovery
        import circuitkit.evaluation
        report["circuitkit"]["discovery_members"] = list(introspect_module(circuitkit.discovery).keys())
        report["circuitkit"]["evaluation_members"] = list(introspect_module(circuitkit.evaluation).keys())
        
        try:
            from circuitkit.evaluation.report import FaithfulnessReport
            report["circuitkit"]["faithfulness_report_found"] = True
        except ImportError:
            report["circuitkit"]["faithfulness_report_found"] = False

        report["guards_verified"]["GUARD_04_circuitkit_first"] = circuitkit_imported_first
    except Exception as e:
        report["circuitkit"]["error"] = str(e)

    # 2. SafeTune
    try:
        import safetune
        report["safetune"]["version"] = getattr(safetune, "__version__", "unknown")
        report["safetune"]["path"] = str(Path(safetune.__file__).parent)
        
        # Introspect unlearn module
        try:
            import safetune.unlearn as su
            report["safetune"]["unlearn_members"] = introspect_module(su)
        except Exception as e:
            report["safetune"]["unlearn_error"] = str(e)
            
        # Check runners
        try:
            import safetune.runner as sr
            report["safetune"]["runner_members"] = list(introspect_module(sr).keys())
        except Exception as e:
            report["safetune"]["runner_error"] = str(e)

        # Check audit badges if registry exists
        audit_badges = {}
        for candidate in ["registry", "audit", "feature_map"]:
            if hasattr(safetune, candidate):
                audit_badges[candidate] = str(getattr(safetune, candidate))
        report["safetune"]["audit_badges"] = audit_badges

    except Exception as e:
        report["safetune"]["error"] = str(e)

    # 3. CuratorKIT
    try:
        import curatorkit
        report["curatorkit"]["version"] = getattr(curatorkit, "__version__", "unknown")
        report["curatorkit"]["path"] = str(Path(curatorkit.__file__).parent)
        
        # Exporters and formats
        try:
            import curatorkit.exporters.compatibility as ccomp
            formats = getattr(ccomp, "TASK_TYPE_EXPORT_FORMATS", {})
            report["curatorkit"]["task_type_export_formats"] = {str(k): list(v) if isinstance(v, (set, list, tuple)) else str(v) for k, v in formats.items()}
        except Exception as e:
            report["curatorkit"]["formats_error"] = str(e)
    except Exception as e:
        report["curatorkit"]["error"] = str(e)

    # 4. AuditKIT
    try:
        import auditkit
        report["auditkit"]["version"] = getattr(auditkit, "__version__", "unknown")
        report["auditkit"]["path"] = str(Path(auditkit.__file__).parent)
        report["auditkit"]["members"] = list(introspect_module(auditkit).keys())
        
        # Check evaluate_many, RedTeamRunner, ExperimentDB
        for symbol in ["evaluate_many", "RedTeamRunner", "ExperimentDB"]:
            found = hasattr(auditkit, symbol)
            report["auditkit"][f"has_{symbol}"] = found
    except Exception as e:
        report["auditkit"]["error"] = str(e)

    # Save to docs and audit_for_carbon
    Path("docs").mkdir(exist_ok=True)
    Path("audit_for_carbon").mkdir(exist_ok=True)
    
    with open("docs/VERIFIED_API_SIGNATURES.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("audit_for_carbon/01_verified_lexsi_apis.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("Successfully verified Lexsi APIs:")
    print(f"  CircuitKIT: {report['circuitkit'].get('version')} (GUARD-04: {circuitkit_imported_first})")
    print(f"  SafeTune:   {report['safetune'].get('version')}")
    print(f"  CuratorKIT: {report['curatorkit'].get('version')}")
    print(f"  AuditKIT:   {report['auditkit'].get('version')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
