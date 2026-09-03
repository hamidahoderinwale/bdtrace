#!/usr/bin/env python3
"""bdtrace: one entry point for the repo's runnable scripts.

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
        "sessiongrep": "export_traces_for_sessiongrep",  # -> sessiongrep-indexable session files
        "fetch": "fetch_all_agent_trajectories",         # full trajectories, outcome-tagged
        "parse": "parse_to_traces",                      # raw Cursor DB text dumps -> traces
    },
    "certs": {
        "extract": "run_extraction_pipeline",       # edit certificates from a benchmark
        "distances": "build_distance_matrices",
        "diversity": "run_diversity_analysis",
    },
}

# noun -> script subdirectory; `bdtrace <noun> <stem>` runs, bare noun lists
DIR_NOUNS = {
    "paper": "agent_trajectories_paper",
    "fig": "figures",
    "analysis": "analysis",
    "tools": "tools",
}

USAGE = """\
usage: bdtrace <object> <action> [args...]   (args go to the script's own argparse; -h works)

  transform list [--examples]
                             enumerate the representation transformations; --examples
                             shows real captured output for each
  transform <name>|all --in records.jsonl [--out F] [--model M] [--limit N] [--llm]
                             apply one transformation, or all of them, to a JSONL of
                             records; `all` includes the LLM-backed ones only with --llm
  config                     model/provider config: which key is set, org-key reachability
  trace import --source claude|cursor|swe_agent|openhands [--input P] [--out F] [--limit N]
                             pull traces out of a local agent store (Claude Code
                             session dir, Cursor SQLite DB, .traj dir) -> JSONL
  trace export --in traces.jsonl --out F
                             re-serialize: .jsonl .jsonl.gz .jsonl.zst .parquet .msgpack
  trace push --in traces.jsonl --repo-id user/name [--public] [--dry-run]
                             push straight to the Hugging Face hub (parquet-backed)
  trace audit --in F [--redact TERM] [--strict]
                             what identity survived: residual classes the anonymizer
                             should have caught, plus candidates only a human can name
  trace spec [--in F]        the record spec; with --in, audit a file (counts, coverage,
                             event types, prompts, text volume, time span, sources)
  trace head --in F [-n N] [--skip N] [--interval I] [--out F]
                             view samples; --out writes the slice as a segmented export
  trace query --in F [--grep R] [--where f=v] [--interval I] [--semantic "..."] [--top-k K]
                             additive filters, embedding-ranked when --semantic; --sort time
                             for chronology, --out to export the matches
  trace index --in F [--model M]
                             build/update the embedding index beside the corpus; unchanged
                             records are never re-embedded, and query then embeds only the query
  trace sessiongrep|fetch|parse
                             sessiongrep session-file export, S3 trajectory fetch,
                             raw Cursor text-dump parse
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

shorthand: the tool name carries the trace noun, so `bdtrace import|export|push` work bare
(also im|ex); tf = transform, cfg = config, nb = notebooks, ls = scripts
"""

# `bdtrace import` == `bdtrace trace import`; two-letter forms for the frequent verbs
VERB_ELISION = {"import": "import", "im": "import", "export": "export", "ex": "export", "push": "push"}
TOP_ALIASES = {"tf": "transform", "cfg": "config", "nb": "notebooks", "ls": "scripts"}

# the core install is tiny; commands needing a heavy dependency name the extra that carries it
EXTRA_FOR_MODULE = {"sentence_transformers": "semantic", "torch": "semantic", "datasets": "hub",
                    "huggingface_hub": "hub", "dspy": "llm", "pandas": "parquet", "pyarrow": "parquet",
                    "numpy": "research", "networkx": "research", "msgpack": "research"}

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
        sys.exit(f"bdtrace: no script {path.relative_to(REPO_ROOT)}")
    sys.argv = [str(path), *args]
    runpy.run_path(str(path), run_name="__main__")


def _run_by_stem(stem: str, args: list[str]) -> None:
    matches = [p for ps in _groups().values() for p in ps if p.stem == stem]
    if not matches:
        sys.exit(f"bdtrace: no script named `{stem}`; `bdtrace scripts {stem}` to search")
    if len(matches) > 1:
        rels = ", ".join(str(p.relative_to(SCRIPTS_DIR)) for p in matches)
        sys.exit(f"bdtrace: `{stem}` is ambiguous ({rels}); use the noun form, e.g. `bdtrace fig <stem>`")
    _exec(matches[0], args)


def _list(filter_: str | None) -> None:
    for group, paths in _groups().items():
        stems = [p.stem for p in paths if not filter_ or filter_ in p.stem]
        if stems:
            print(f"{group or 'top level (bdtrace run <stem>)'}:")
            print("".join(f"  {s}\n" for s in stems), end="")


def _notebooks(args: list[str]) -> None:
    dataset = "swe_bench_lite"
    if args[:1] == ["--datasets"] and len(args) == 2:
        dataset = args[1]
    elif args:
        sys.exit("bdtrace notebooks takes only --datasets <name>")
    for stem, template in LEGACY_CHAIN:
        stage_args = [a.format(ds=dataset) for a in template]
        print(f"== {stem} {' '.join(stage_args)}")
        # subprocess per stage so one script's globals/argv can't leak into the next
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / f"{stem}.py"), *stage_args])
        if result.returncode != 0:
            sys.exit(f"bdtrace: `{stem}` failed (exit {result.returncode}); fix and re-run from it")


def _trace(rest: list[str]) -> None:
    """import/export/push are module-backed (work from a bare install); the rest are scripts."""
    import argparse

    verb, rest = (rest[0], rest[1:]) if rest else ("", [])
    if verb == "import":
        p = argparse.ArgumentParser(prog="bdtrace trace import",
                                    description="Pull traces out of a local agent store, standardized to JSONL")
        p.add_argument("--source", choices=["claude", "cursor", "swe_agent", "openhands"], default="claude")
        p.add_argument("--input", type=Path, default=None, help="store path (default: the source's standard location)")
        p.add_argument("--out", type=Path, default=Path("traces.jsonl"))
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--agent", default=None,
                       help="agent label carried on every record (default: the source name); "
                            "procgrep groups and compares by this field")
        a = p.parse_args(rest)
        import json

        if a.source == "claude":
            from analysis.ingest.claude_code import iter_traces
            records = iter_traces(a.input, limit=a.limit)
        else:
            from analysis.ingest.harnesses import parse
            store = a.input
            if store is None and a.source == "cursor":
                store = Path.home() / "Library/Application Support/Cursor/User"
            if store is None:
                sys.exit(f"bdtrace: --input is required for --source {a.source} (no standard local location)")
            records = parse(a.source, store, limit=a.limit)
        n = 0
        # stdout is for data (clig.dev): --out - streams JSONL for piping, status goes to stderr
        f = sys.stdout if str(a.out) == "-" else open(a.out, "w")
        agent = a.agent or a.source
        for r in records:
            f.write(json.dumps({**r, "agent": r.get("agent") or agent}, default=str) + "\n")
            n += 1
        if f is not sys.stdout:
            f.close()
        print(f"{n} traces -> {a.out}", file=sys.stderr)
    elif verb == "export":
        p = argparse.ArgumentParser(prog="bdtrace trace export",
                                    description="Re-serialize a trace JSONL; format inferred from --out suffix")
        p.add_argument("--in", dest="in_path", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True,
                       help=".jsonl | .jsonl.gz | .jsonl.zst | .parquet | .msgpack")
        p.add_argument("--types", default=None,
                       help="event types to keep: comma list from the taxonomy, or 'tools' (= all but prompt)")
        p.add_argument("--compact", action="store_true",
                       help="reduce each event to its action surface (text bodies dropped)")
        p.add_argument("--anonymize", action="store_true",
                       help="strip home dirs, usernames in paths, and emails from all strings")
        p.add_argument("--redact", action="append", default=[], metavar="TERM",
                       help="also scrub this literal term; repeatable. `trace audit` names candidates")
        p.add_argument("--strict", action="store_true",
                       help="with --anonymize: refuse to keep the output if any identity survives")
        p.add_argument("--interval", default=None,
                       help="event-time window, as in trace head (A..B, or 7d/24h/2w)")
        a = p.parse_args(rest)
        from bdtrace import spec
        from bdtrace.export import export_traces
        records = a.in_path
        redact = tuple(a.redact)
        if a.types or a.compact or a.anonymize or a.interval:
            since, until = spec.interval_bounds(a.interval)
            types = spec.parse_types(a.types) if a.types else None
            def shaped():
                for r in spec._iter_records(a.in_path):
                    if not spec._in_window(r, since, until):
                        continue
                    r = spec.project(r, types, a.compact)
                    if a.anonymize:
                        r = spec.anonymize(r, redact)
                    yield r
            records = shaped()
        out = export_traces(records, a.out)
        from bdtrace.meta import write_sidecar
        sidecar = write_sidecar(out, a.in_path, {"types": a.types, "compact": a.compact,
                                                 "anonymize": a.anonymize, "interval": a.interval,
                                                 "redact": list(redact)})
        print(f"{out} ({out.stat().st_size:,} bytes) + {sidecar.name}")
        # an anonymized artifact is audited before it is called done; --strict refuses
        # to leave a leaking file on disk rather than reporting the leak after the fact
        if a.anonymize and out.suffix == ".jsonl":
            from bdtrace.audit import audit, report
            result = audit(out, redact)
            if result["residual"]:
                print(report(result), file=sys.stderr)
                if a.strict:
                    out.unlink()
                    sidecar.unlink()
                    sys.exit(f"bdtrace: identity survived anonymization; {out.name} removed (--strict)")
            elif a.strict:
                print("audit: no residual identity", file=sys.stderr)
    elif verb == "push":
        p = argparse.ArgumentParser(prog="bdtrace trace push",
                                    description="Push a trace JSONL to the Hugging Face hub (parquet-backed dataset)")
        p.add_argument("--in", dest="in_path", type=Path, required=True)
        p.add_argument("--repo-id", required=True, help="e.g. midah/my-traces")
        p.add_argument("--public", action="store_true", help="default is a private dataset")
        p.add_argument("--dry-run", action="store_true", help="build the dataset, report rows, no upload")
        p.add_argument("--dataset-version", default="1.0.0", help="semver carried in the Croissant (data grew = patch, shape changed = minor, meaning changed = major)")
        a = p.parse_args(rest)
        from bdtrace.export import push_traces
        from bdtrace.meta import build_croissant, push_croissant
        print(push_traces(a.in_path, a.repo_id, private=not a.public, dry_run=a.dry_run))
        with open(a.in_path) as f:
            n_rows = sum(1 for line in f if line.strip())
        croissant = build_croissant(a.repo_id, n_rows, f"local trace JSONL {a.in_path.name}", a.dataset_version)
        if a.dry_run:
            print(f"dry-run: croissant.json would carry version {a.dataset_version}, {n_rows} rows", file=sys.stderr)
        else:
            print(push_croissant(a.repo_id, croissant), file=sys.stderr)
    elif verb == "query":
        p = argparse.ArgumentParser(prog="bdtrace trace query",
                                    description="Additive filters over trace JSONL, optionally semantic-ranked; "
                                                "matches print one line each (score, id, start time, prompt head)")
        p.add_argument("--in", dest="in_path", type=Path, required=True)
        p.add_argument("--grep", default=None, help="case-insensitive regex over the record's text")
        p.add_argument("--where", action="append", default=[], help="field=value equality; repeatable (ANDed)")
        p.add_argument("--interval", default=None, help="event-time window: A..B, or 7d/24h/2w")
        p.add_argument("--semantic", default=None, help="embedding query; ranks survivors by cosine")
        p.add_argument("--top-k", type=int, default=20, help="matches kept when --semantic ranks (default 20)")
        p.add_argument("--min-score", type=float, default=None, help="cosine floor; replaces the top-k cut")
        p.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
        p.add_argument("--limit", type=int, default=None, help="cap on records read")
        p.add_argument("--sort", choices=["score", "time"], default="score",
                       help="order matches by rank (default) or chronologically by first event")
        p.add_argument("--out", type=Path, default=None, help="also write matching records as JSONL ('-' = stdout)")
        a = p.parse_args(rest)
        import json as _json

        from bdtrace.query import query
        matches = list(query(a.in_path, grep=a.grep, where=a.where, interval=a.interval,
                             semantic=a.semantic, top_k=a.top_k, min_score=a.min_score,
                             limit=a.limit, model=a.model))
        def start_ts(r):
            return min((e.get("timestamp") or "~" for e in r.get("events", [])), default="~")
        if a.sort == "time":
            matches.sort(key=lambda m: start_ts(m[0]))
        for r, score in matches:
            prompt = (r.get("prompts") or [{}])[0].get("text", "")[:80].replace("\n", " ")
            line = f"{'' if score is None else f'{score:.3f}  '}{r.get('instance_id')}  {start_ts(r)[:19]}  {prompt}"
            print(line if a.out else line, file=sys.stderr if a.out else sys.stdout)
        if a.out:
            f = sys.stdout if str(a.out) == "-" else open(a.out, "w")
            for r, _ in matches:
                f.write(_json.dumps(r, default=str) + "\n")
            if f is not sys.stdout:
                f.close()
        print(f"{len(matches)} matches", file=sys.stderr)
    elif verb == "index":
        p = argparse.ArgumentParser(prog="bdtrace trace index",
                                    description="Build or update the embedding index beside a trace JSONL; "
                                                "unchanged records are never re-embedded")
        p.add_argument("--in", dest="in_path", type=Path, required=True)
        p.add_argument("--model", default=None, help="sentence-transformers model (default: all-MiniLM-L6-v2)")
        a = p.parse_args(rest)
        from bdtrace.index import build_index
        info = build_index(a.in_path, a.model)
        print(f"{info['n']} vectors, {info['dims']} dims, model {info['model']}")
    elif verb == "audit":
        p = argparse.ArgumentParser(prog="bdtrace trace audit",
                                    description="What identity is still in a file you are about to share")
        p.add_argument("--in", dest="in_path", type=Path, required=True)
        p.add_argument("--redact", action="append", default=[], metavar="TERM",
                       help="also count occurrences of this literal term; repeatable")
        p.add_argument("--strict", action="store_true", help="exit non-zero when anything residual is found")
        a = p.parse_args(rest)
        from bdtrace.audit import audit, report
        result = audit(a.in_path, tuple(a.redact))
        print(report(result))
        if a.strict and result["residual"]:
            sys.exit(1)
    elif verb == "spec":
        p = argparse.ArgumentParser(prog="bdtrace trace spec",
                                    description="Canonical trace-record spec; with --in, audit a file against it")
        p.add_argument("--in", dest="in_path", type=Path, default=None)
        a = p.parse_args(rest)
        from bdtrace import spec
        print(spec.describe(a.in_path) if a.in_path else spec.spec_text())
    elif verb == "head":
        p = argparse.ArgumentParser(prog="bdtrace trace head",
                                    description="Show the first records legibly; --out writes the slice as JSONL")
        p.add_argument("--in", dest="in_path", type=Path, required=True)
        p.add_argument("-n", type=int, default=3)
        p.add_argument("--skip", type=int, default=0, help="records to skip first (segmented export)")
        p.add_argument("--events", type=int, default=5, help="events shown per record in the view")
        p.add_argument("--out", type=Path, default=None, help="write the slice untruncated instead of printing")
        p.add_argument("--interval", default=None,
                       help="event-time window: 2026-08-01..2026-09-01 (either side open), or 7d / 24h / 2w")
        a = p.parse_args(rest)
        from bdtrace import spec
        spec.head(a.in_path, a.n, a.skip, a.events, a.out, *spec.interval_bounds(a.interval))
    elif verb in COMMANDS["trace"]:
        _exec(SCRIPTS_DIR / f"{COMMANDS['trace'][verb]}.py", rest)
    else:
        sys.exit(f"usage: bdtrace trace import|export|push|spec|head|{'|'.join(COMMANDS['trace'])} [args...]")


def _transform(rest: list[str]) -> None:
    import argparse

    from bdtrace import transforms

    if rest[:1] == ["list"]:
        print(transforms.list_table(examples="--examples" in rest or "-e" in rest))
        return
    parser = argparse.ArgumentParser(prog="bdtrace transform",
                                     description="Apply representation transformations to a JSONL of records")
    parser.add_argument("name", help="a transformation from `bdtrace transform list`, or `all`")
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
        sys.exit(f"bdtrace: unknown transformation `{a.name}`; `bdtrace transform list` enumerates them")
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
    if args[0] in ("-V", "--version"):
        from importlib.metadata import version
        print(version("bdtrace"))
        return
    cmd, rest = args[0], args[1:]
    cmd = TOP_ALIASES.get(cmd, cmd)
    try:
        _dispatch_command(cmd, rest)
    except ImportError as e:
        extra = EXTRA_FOR_MODULE.get((e.name or "").split(".")[0])
        if not extra:
            raise
        sys.exit(f"bdtrace: `{cmd}` needs the `{extra}` extra: uv sync --extra {extra}, or\n"
                 f"  uv tool install 'bdtrace[{extra}] @ git+https://github.com/hamidahoderinwale/bdtrace'")


def _dispatch_command(cmd: str, rest: list[str]) -> None:
    if cmd in VERB_ELISION:
        _trace([VERB_ELISION[cmd], *rest])
    elif cmd == "trace":
        _trace(rest)
    elif cmd == "transform":
        _transform(rest)
    elif cmd == "config":
        from bdtrace.transforms import config_report
        print(config_report())
    elif cmd in COMMANDS:
        verbs = COMMANDS[cmd]
        if not rest or rest[0] not in verbs:
            sys.exit(f"usage: bdtrace {cmd} {'|'.join(verbs)} [args...]")
        _exec(SCRIPTS_DIR / f"{verbs[rest[0]]}.py", rest[1:])
    elif cmd in DIR_NOUNS:
        if not rest:
            _list_dir = _groups()[cmd]
            print("".join(f"{p.stem}\n" for p in _list_dir), end="")
        else:
            _exec(SCRIPTS_DIR / DIR_NOUNS[cmd] / f"{rest[0]}.py", rest[1:])
    elif cmd == "run":
        if not rest:
            sys.exit("usage: bdtrace run <stem> [args...]")
        _run_by_stem(rest[0], rest[1:])
    elif cmd == "scripts":
        _list(rest[0] if rest else None)
    elif cmd == "notebooks":
        _notebooks(rest)
    elif cmd == "lab":
        sys.exit(subprocess.run(["jupyter", "lab"], cwd=REPO_ROOT / "notebooks").returncode)
    else:
        sys.exit(f"bdtrace: unknown command `{cmd}`\n\n{USAGE}")


if __name__ == "__main__":
    main()
