"""Metadata beside exported trace data.

Placement follows the org dataset-versioning rule: JSON Schema is the validation
authority and ships everywhere; a lightweight provenance sidecar rides every
export; Croissant JSON-LD (1.1, with PROV-O) is generated only at the publish
boundary (a hub push), where a consumer machine acts on it.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TRACE_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "bdtrace trace record",
    "type": "object",
    "required": ["instance_id", "events"],
    "properties": {
        "instance_id": {"type": "string"},
        "repo": {"type": ["string", "null"]},
        "base_commit": {"type": ["string", "null"]},
        "cwd": {"type": ["string", "null"]},
        "events": {"type": "array", "items": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"enum": ["prompt", "edit", "read", "search", "run", "test", "other", "code_change"]},
                "timestamp": {"type": ["string", "null"]},
                "details": {"type": "object"},
            },
        }},
        "prompts": {"type": "array"},
        "reprs": {"type": "object"},
    },
}


def _describe_file(path: Path) -> dict:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if path.suffix == ".jsonl":
        with open(path) as f:
            n = sum(1 for line in f if line.strip())
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest(),
            **({"n_records": n} if path.suffix == ".jsonl" else {})}


def write_sidecar(out_path: Path, source: Path, params: dict) -> Path:
    """<out>.meta.json: what these bytes are, where they came from, how they were shaped."""
    from importlib.metadata import version

    from bdtrace import spec
    # the summary is the same structure `trace spec --in` renders; computed on the
    # export itself when it is plain jsonl, else on the source (marked as such)
    if out_path.suffix == ".jsonl":
        summary, summary_of = spec.summarize(out_path), "artifact"
    else:
        summary, summary_of = spec.summarize(source), "source (pre-projection)"
    meta = {
        "artifact": _describe_file(out_path),
        "derived_from": str(source),
        "projection": {k: v for k, v in params.items() if v not in (None, False)},
        "summary": summary,
        "summary_of": summary_of,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": f"bdtrace {version('bidirect-align-dev-traces')}",
        "record_schema": TRACE_JSON_SCHEMA,
    }
    sidecar = out_path.with_name(out_path.name + ".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    return sidecar


def build_croissant(repo_id: str, n_rows: int, source_desc: str, dataset_version: str) -> dict:
    """Croissant 1.1 with PROV-O, for the hub publish. One document per version."""
    from importlib.metadata import version
    fields = [
        {"@type": "cr:Field", "name": f, "description": d,
         "dataType": "sc:Text" if t == "text" else "sc:Boolean" if t == "bool" else "sc:Text"}
        for f, t, d in [
            ("instance_id", "text", "unique trace id (benchmark instance, or <harness>-<session id>)"),
            ("repo", "text", "repository worked on; null for session-derived traces"),
            ("base_commit", "text", "commit the work started from; null for session-derived traces"),
            ("events", "text", "JSON-encoded ordered action list: {type: prompt|edit|read|search|run|test|other, details, timestamp}"),
            ("prompts", "text", "JSON-encoded prompt events, extracted for convenience"),
        ]
    ]
    return {
        "@context": {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/",
                     "sc": "https://schema.org/", "prov": "http://www.w3.org/ns/prov#",
                     "dct": "http://purl.org/dc/terms/"},
        "@type": "sc:Dataset",
        "dct:conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": repo_id.split("/")[-1],
        "description": f"Developer workflow traces in the bdtrace standardized record shape; {source_desc}. "
                       "Nested fields are JSON-encoded string columns in the parquet distribution.",
        "version": dataset_version,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "recordSet": [{"@type": "cr:RecordSet", "name": "traces", "field": fields,
                       "description": f"{n_rows} trace records"}],
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "prov:used": source_desc,
            "prov:wasAssociatedWith": {"@type": "prov:SoftwareAgent",
                                       "name": f"bdtrace {version('bidirect-align-dev-traces')}"},
            "prov:endedAtTime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def push_croissant(repo_id: str, croissant: dict) -> str:
    """Upload croissant.json beside the pushed dataset, same token ladder as the push."""
    import io

    from huggingface_hub import HfApi

    from bdtrace.export import resolve_hf_token
    api = HfApi(token=resolve_hf_token()[0])
    api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(croissant, indent=2).encode()),
                    path_in_repo="croissant.json", repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/croissant.json"
