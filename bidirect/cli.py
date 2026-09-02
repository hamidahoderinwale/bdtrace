#!/usr/bin/env python3
"""bidirect: one entry point for the repo's runnable scripts.

Shape: noun-verb commands over the repo's objects (traces, certificates,
figures, paper numbers), each backed by an existing script; directory nouns
expose a whole script group; `run` reaches anything else by stem. No logic
lives here: dispatch is runpy over the script file, so each script's own
argparse interface is unchanged and heavy imports happen only for the script
actually invoked.
"""

import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# noun -> verb -> script (relative to scripts/)
COMMANDS = {
    "trace": {
        "export": "export_traces_for_sessiongrep",  # -> sessiongrep-indexable session files
        "fetch": "fetch_all_agent_trajectories",    # full trajectories, outcome-tagged
        "parse": "parse_to_traces",                 # raw Cursor DB exports -> traces
    },
    "certs": {
        "extract": "run_extraction_pipeline",       # edit certificates from a benchmark
        "distances": "build_distance_matrices",
        "diversity": "run_diversity_analysis",
    },
}

# noun -> script subdirectory; `bidirect <noun> <stem>` runs, bare noun lists
DIR_NOUNS = {
    "paper": "agent_trajectories_paper",
    "fig": "figures",
    "analysis": "analysis",
    "tools": "tools",
}

USAGE = """\
usage: bidirect <object> <action> [args...]   (args go to the script's own argparse; -h works)

  transform list             enumerate the representation transformations
  transform <name>|all --in records.jsonl [--out F] [--model M] [--limit N] [--llm]
                             apply one transformation, or all of them, to a JSONL of
                             records; `all` includes the LLM-backed ones only with --llm
  config                     model/provider config: which key is set, org-key reachability
  trace export|fetch|parse   dev traces: export to sessiongrep session files,
                             fetch trajectories from S3, parse Cursor DB dumps
  certs extract|distances|diversity   edit certificates and the measures over them
  paper [<script>]           paper figure/number scripts (bare = list, with grounding
                             status in scripts/agent_trajectories_paper/README.md)
  fig [<script>]             figure scripts (bare = list)
  analysis [<script>]        analysis scripts (bare = list)
  tools [<script>]           tools (procgrep search surface)
  notebooks [--datasets D]   regenerate the exploration notebooks' inputs via the
                             legacy first-gen chain (default swe_bench_lite)
  lab                        JupyterLab in notebooks/
  run <stem> [args...]       any other script by stem, searched across scripts/
  scripts [filter]           list every runnable script
"""

# the legacy first-gen chain the exploration notebooks read (notebooks/README.md)
LEGACY_CHAIN = [
    ("run_extraction_pipeline", ["--datasets", "{ds}", "--output-dir", "output"]),
    ("build_distance_matrices", ["--input", "output/datasets/{ds}/test.parquet",
                                 "--reprs", "tokens", "edits_set_diff", "edits_tree", "modules_graph"]),
    ("run_diversity_analysis", ["--matrices", "output/datasets/{ds}/distances.parquet",
                                "--labels", "output/datasets/{ds}/labels.parquet"]),
    ("run_multi_benchmark_plots", ["--benchmarks", "{ds}"]),
]


def _groups() -> dict[str, list[Path]]:
    groups = {"": sorted(SCRIPTS_DIR.glob("*.py"))}
    for noun, subdir in DIR_NOUNS.items():
        groups[noun] = sorted((SCRIPTS_DIR / subdir).glob("*.py"))
    return {g: [p for p in ps if not p.stem.startswith("_")] for g, ps in groups.items()}


def _exec(path: Path, args: list[str]) -> None:
    if not path.is_file():
        sys.exit(f"bidirect: no script {path.relative_to(REPO_ROOT)}")
    sys.argv = [str(path), *args]
    runpy.run_path(str(path), run_name="__main__")


def _run_by_stem(stem: str, args: list[str]) -> None:
    matches = [p for ps in _groups().values() for p in ps if p.stem == stem]
    if not matches:
        sys.exit(f"bidirect: no script named `{stem}`; `bidirect scripts {stem}` to search")
    if len(matches) > 1:
        rels = ", ".join(str(p.relative_to(SCRIPTS_DIR)) for p in matches)
        sys.exit(f"bidirect: `{stem}` is ambiguous ({rels}); use the noun form, e.g. `bidirect fig <stem>`")
    _exec(matches[0], args)


def _list(filter_: str | None) -> None:
    for group, paths in _groups().items():
        stems = [p.stem for p in paths if not filter_ or filter_ in p.stem]
        if stems:
            print(f"{group or 'top level (bidirect run <stem>)'}:")
            print("".join(f"  {s}\n" for s in stems), end="")


def _notebooks(args: list[str]) -> None:
    dataset = "swe_bench_lite"
    if args[:1] == ["--datasets"] and len(args) == 2:
        dataset = args[1]
    elif args:
        sys.exit("bidirect notebooks takes only --datasets <name>")
    for stem, template in LEGACY_CHAIN:
        stage_args = [a.format(ds=dataset) for a in template]
        print(f"== {stem} {' '.join(stage_args)}")
        # subprocess per stage so one script's globals/argv can't leak into the next
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / f"{stem}.py"), *stage_args])
        if result.returncode != 0:
            sys.exit(f"bidirect: `{stem}` failed (exit {result.returncode}); fix and re-run from it")


def _transform(rest: list[str]) -> None:
    import argparse

    from bidirect import transforms

    if rest[:1] == ["list"]:
        print(transforms.list_table())
        return
    parser = argparse.ArgumentParser(prog="bidirect transform",
                                     description="Apply representation transformations to a JSONL of records")
    parser.add_argument("name", help="a transformation from `bidirect transform list`, or `all`")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="input JSONL")
    parser.add_argument("--out", type=Path, default=None, help="output JSONL (default: <in>.reprs.jsonl)")
    parser.add_argument("--model", default=None, help="LM for inferred transforms (default: env BIDIRECT_MODEL)")
    parser.add_argument("--limit", type=int, default=None, help="process at most N records")
    parser.add_argument("--llm", action="store_true", help="with `all`: include the LLM-backed transforms")
    parser.add_argument("--before-field", default="before", help="patch transforms: before-source field")
    parser.add_argument("--after-field", default="after", help="patch transforms: after-source field")
    a = parser.parse_args(rest)
    if a.name == "all":
        names = [n for n, t in transforms.TRANSFORMS.items() if a.llm or not t.llm]
    elif a.name in transforms.TRANSFORMS:
        names = [a.name]
    else:
        sys.exit(f"bidirect: unknown transformation `{a.name}`; `bidirect transform list` enumerates them")
    if any(transforms.TRANSFORMS[n].llm for n in names):
        model = transforms.configure_llm(a.model)
        print(f"inferred transforms via {model}", file=sys.stderr)
    out = a.out or a.in_path.with_suffix(".reprs.jsonl")
    transforms.apply(names, a.in_path, out, a.before_field, a.after_field, a.limit)


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return
    cmd, rest = args[0], args[1:]
    if cmd == "transform":
        _transform(rest)
    elif cmd == "config":
        from bidirect.transforms import config_report
        print(config_report())
    elif cmd in COMMANDS:
        verbs = COMMANDS[cmd]
        if not rest or rest[0] not in verbs:
            sys.exit(f"usage: bidirect {cmd} {'|'.join(verbs)} [args...]")
        _exec(SCRIPTS_DIR / f"{verbs[rest[0]]}.py", rest[1:])
    elif cmd in DIR_NOUNS:
        if not rest:
            _list_dir = _groups()[cmd]
            print("".join(f"{p.stem}\n" for p in _list_dir), end="")
        else:
            _exec(SCRIPTS_DIR / DIR_NOUNS[cmd] / f"{rest[0]}.py", rest[1:])
    elif cmd == "run":
        if not rest:
            sys.exit("usage: bidirect run <stem> [args...]")
        _run_by_stem(rest[0], rest[1:])
    elif cmd == "scripts":
        _list(rest[0] if rest else None)
    elif cmd == "notebooks":
        _notebooks(rest)
    elif cmd == "lab":
        sys.exit(subprocess.run(["jupyter", "lab"], cwd=REPO_ROOT / "notebooks").returncode)
    else:
        sys.exit(f"bidirect: unknown command `{cmd}`\n\n{USAGE}")


if __name__ == "__main__":
    main()
