#!/usr/bin/env python3
"""bidirect: one entry point for the repo's pipeline scripts.

Named subcommands cover the documented certificate pipeline; `run` reaches
any other script in scripts/ by stem. Dispatch is runpy over the script
files, so each script's own argparse interface is unchanged and heavy
imports only happen for the subcommand actually invoked.
"""

import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# The documented pipeline (notebooks/README.md), in stage order.
PIPELINE = {
    "extract": "run_extraction_pipeline",
    "distances": "build_distance_matrices",
    "diversity": "run_diversity_analysis",
    "plots": "run_multi_benchmark_plots",
}

USAGE = """\
usage: bidirect <command> [args...]

Pipeline (each forwards args to the underlying script; -h for its options):
  extract      run_extraction_pipeline.py   certificates from a benchmark
  distances    build_distance_matrices.py   pairwise distances per representation
  diversity    run_diversity_analysis.py    diversity metrics + rank correlations
  plots        run_multi_benchmark_plots.py distributional plots

Composites:
  notebooks [--datasets D]  regenerate the notebook inputs end to end
                            (extract -> distances -> diversity -> plots,
                            default dataset swe_bench_lite)
  lab                       launch JupyterLab in notebooks/

Everything else:
  run <script> [args...]    any scripts/*.py by stem, e.g. `bidirect run backbone_probe -h`
  scripts                   list available script stems
"""


def _script_path(stem: str) -> Path:
    path = SCRIPTS_DIR / f"{stem}.py"
    if not path.is_file():
        sys.exit(f"bidirect: no script {path.relative_to(REPO_ROOT)}; `bidirect scripts` lists them")
    return path


def _dispatch(stem: str, args: list[str]) -> None:
    path = _script_path(stem)
    sys.argv = [str(path), *args]
    runpy.run_path(str(path), run_name="__main__")


def _notebooks(args: list[str]) -> None:
    dataset = "swe_bench_lite"
    if args[:1] == ["--datasets"] and len(args) == 2:
        dataset = args[1]
    elif args:
        sys.exit("bidirect notebooks takes only --datasets <name>")
    out = Path("output") / "datasets" / dataset
    stages = [
        ("extract", ["--datasets", dataset, "--output-dir", "output"]),
        ("distances", ["--input", str(out / "test.parquet"),
                       "--reprs", "tokens", "edits_set_diff", "edits_tree", "modules_graph"]),
        ("diversity", ["--matrices", str(out / "distances.parquet"),
                       "--labels", str(out / "labels.parquet")]),
        ("plots", ["--benchmarks", dataset]),
    ]
    for name, stage_args in stages:
        print(f"== bidirect {name} {' '.join(stage_args)}")
        # each stage runs in a subprocess so one script's globals/argv can't leak into the next
        result = subprocess.run([sys.executable, str(_script_path(PIPELINE[name])), *stage_args])
        if result.returncode != 0:
            sys.exit(f"bidirect: stage `{name}` failed (exit {result.returncode}); fix and re-run from it")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return
    cmd, rest = args[0], args[1:]
    if cmd in PIPELINE:
        _dispatch(PIPELINE[cmd], rest)
    elif cmd == "run":
        if not rest:
            sys.exit("usage: bidirect run <script> [args...]")
        _dispatch(rest[0], rest[1:])
    elif cmd == "scripts":
        for p in sorted(SCRIPTS_DIR.glob("*.py")):
            print(p.stem)
    elif cmd == "notebooks":
        _notebooks(rest)
    elif cmd == "lab":
        sys.exit(subprocess.run(["jupyter", "lab"], cwd=REPO_ROOT / "notebooks").returncode)
    else:
        sys.exit(f"bidirect: unknown command `{cmd}`\n\n{USAGE}")


if __name__ == "__main__":
    main()
