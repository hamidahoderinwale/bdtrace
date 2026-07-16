#!/usr/bin/env python3
"""
Build scoped certificates for all 300 oracle instances.

Reads resolved_traces_lite_full.jsonl, computes scoped certs with
file-level and scope-level information, and saves to
output/scoped_certificates/oracle_scoped_certs.json.

Usage:
    uv run python scripts/build_scoped_certificates.py
"""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load modules manually to bypass the heavy __init__.py chain in
# analysis.procedures (which pulls numpy etc via procedure_divergence).
def _load_mod(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_m1 = _load_mod("ast_edit_sequences", "analysis/procedures/ast_edit_sequences.py")
_m2 = _load_mod("contextual_edit_ops", "analysis/procedures/contextual_edit_ops.py")
if "analysis.procedures" not in sys.modules or not hasattr(sys.modules["analysis.procedures"], "__path__"):
    _ap = type(sys)("analysis.procedures")
    _ap.__path__ = [str(ROOT / "analysis" / "procedures")]
    sys.modules["analysis.procedures"] = _ap
if "analysis" not in sys.modules or not hasattr(sys.modules["analysis"], "__path__"):
    _a = type(sys)("analysis")
    _a.__path__ = [str(ROOT / "analysis")]
    sys.modules["analysis"] = _a
sys.modules["analysis.procedures.ast_edit_sequences"] = _m1
sys.modules["analysis.procedures.contextual_edit_ops"] = _m2

from analysis.procedures.scoped_edit_ops import trace_to_scoped_cert

TRACES_PATH = ROOT / "output" / "resolved_traces_lite_full.jsonl"
OUTPUT_DIR = ROOT / "output" / "scoped_certificates"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading oracle traces...")
    traces = []
    with open(TRACES_PATH) as f:
        for line in f:
            traces.append(json.loads(line))
    print(f"  {len(traces)} traces loaded")

    print("\nComputing scoped certificates...")
    certs = []
    failed = 0
    for i, trace in enumerate(traces):
        cert = trace_to_scoped_cert(trace)
        if cert is None:
            failed += 1
            continue
        certs.append(cert)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(traces)} processed")

    print(f"  {len(certs)} certificates computed ({failed} failed)")

    # Save
    out_path = OUTPUT_DIR / "oracle_scoped_certs.json"
    with open(out_path, "w") as f:
        json.dump(certs, f, indent=2)
    print(f"\nSaved {out_path.relative_to(ROOT)}")

    # Summary stats
    print("\n--- Summary ---")

    # Patch size
    sizes = [c["patch_size"] for c in certs]
    print(f"Patch size: mean={sum(sizes)/len(sizes):.1f}, "
          f"median={sorted(sizes)[len(sizes)//2]}, "
          f"min={min(sizes)}, max={max(sizes)}")

    # Scope count
    scope_counts = [len(c["scopes_touched"]) for c in certs]
    scope_counter = Counter(scope_counts)
    print(f"\nScope count distribution:")
    for count in sorted(scope_counter.keys()):
        print(f"  {count} scopes: {scope_counter[count]} instances")

    # Module distribution
    module_counter = Counter(c["file_module"] for c in certs)
    print(f"\nTop 15 modules:")
    for mod, count in module_counter.most_common(15):
        print(f"  {mod}: {count}")

    # Hunk count
    hunk_counts = [c["hunk_count"] for c in certs]
    hunk_counter = Counter(hunk_counts)
    print(f"\nHunk count distribution:")
    for count in sorted(hunk_counter.keys()):
        print(f"  {count} hunks: {hunk_counter[count]} instances")

    # Scope types
    type_counter = Counter()
    for c in certs:
        type_counter.update(c["scope_types"])
    print(f"\nScope type frequency:")
    for st, count in type_counter.most_common():
        print(f"  {st}: {count}")

    # Edit cert size
    cert_sizes = [len(c["edit_cert"]) for c in certs]
    print(f"\nEdit cert size: mean={sum(cert_sizes)/len(cert_sizes):.1f}, "
          f"median={sorted(cert_sizes)[len(cert_sizes)//2]}, "
          f"min={min(cert_sizes)}, max={max(cert_sizes)}")


if __name__ == "__main__":
    main()
