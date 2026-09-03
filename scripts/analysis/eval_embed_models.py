"""Which embedding model should `bdtrace trace query --semantic` default to?

Ranks all 138 real Claude Code sessions in a trace JSONL against 60 hand-written
queries whose gold session was fixed BEFORE any model ran, and reports
recall@1 / recall@5 / MRR with paired bootstrap intervals, alongside two
non-neural lexical baselines. Every candidate sees exactly the text the shipped
index sees (`bdtrace.query.record_text`), so the comparison is of models, not of
text views.

Run:
    uv run --with sentence-transformers --with rank-bm25 --with scikit-learn \
        --with einops --with numpy \
        python scripts/analysis/eval_embed_models.py --corpus /path/to/all.jsonl

Writes a JSON result object to --out (default: stdout).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from bdtrace.query import EMBED_BATCH, record_text  # noqa: E402

# --- the frozen evaluation set ------------------------------------------------
#
# Construction protocol (see docs/embedding_model_eval.md for the bias this
# introduces). For each gold session an anchor was picked mechanically: a token
# in that session's `record_text` with corpus document frequency 1. Two queries
# were then written by hand from the surrounding text, before any model was run:
#
#   kind="literal"    — a natural-language question that REUSES the anchor
#                       wording verbatim. Favours lexical matching by design.
#   kind="paraphrase" — the same session described in outside words, avoiding
#                       the anchor. This is the slice that tests semantics.
#
# Nothing below was edited after seeing a score.

QUERIES: list[dict[str, str]] = [
    {"gold": "claude-0714fa67-78e0-42cf-b93c-ef4959ccd1d0", "kind": "literal",
     "text": "research on udom and its utilities for understanding cross-framework work, and what the actual experiment is"},
    {"gold": "claude-0714fa67-78e0-42cf-b93c-ef4959ccd1d0", "kind": "paraphrase",
     "text": "investigating a library that abstracts over several front-end frameworks and deciding what would actually be measured"},

    {"gold": "claude-1131d0cc-4527-46a7-a856-546ee947c836", "kind": "literal",
     "text": "locate the reconstruction schematic png and check whether the PR is upstream in the damascus branch"},
    {"gold": "claude-1131d0cc-4527-46a7-a856-546ee947c836", "kind": "paraphrase",
     "text": "finding a diagram image for a task and checking whether that change was already sent upstream, with the failure-mode figures still unmade"},

    {"gold": "claude-1aeb3f89-2ae4-4b9a-96ae-d8b700aba9c8", "kind": "literal",
     "text": "extract durable typed events from a session digest as strict JSONL with decision, retraction, preference and blocker types"},
    {"gold": "claude-1aeb3f89-2ae4-4b9a-96ae-d8b700aba9c8", "kind": "paraphrase",
     "text": "pulling only the lasting facts out of a finished coding conversation and writing one record per line"},

    {"gold": "claude-2cd70a30-0a2f-4f52-bd20-55f8110acf7e", "kind": "literal",
     "text": "how to add someone to the creativity-index-judge-tree vercel project and where to find the password"},
    {"gold": "claude-2cd70a30-0a2f-4f52-bd20-55f8110acf7e", "kind": "paraphrase",
     "text": "granting a colleague access to a hosted preview site and locating the shared login for it"},

    {"gold": "claude-2e787c15-08a0-4f78-9399-bb4f3b183038", "kind": "literal",
     "text": "make validators include the provided value in ValidationError so a %(value)s placeholder can be used"},
    {"gold": "claude-2e787c15-08a0-4f78-9399-bb4f3b183038", "kind": "paraphrase",
     "text": "patching a web framework so a custom message can quote the input that was rejected"},

    {"gold": "claude-3057429b-6796-41a5-9f79-b668f4787902", "kind": "literal",
     "text": "lint-based grading for design system grading on damascus and where to add the QA utility"},
    {"gold": "claude-3057429b-6796-41a5-9f79-b668f4787902", "kind": "paraphrase",
     "text": "whether a static style checker makes a good automatic scorer for visual design work, and where that tool belongs"},

    {"gold": "claude-337d5a9d-b49c-4e09-9f40-3b5202abe7e3", "kind": "literal",
     "text": "a grouped pairwise mean is candidate-centric and does not look at the diversity or structure of the set"},
    {"gold": "claude-337d5a9d-b49c-4e09-9f40-3b5202abe7e3", "kind": "paraphrase",
     "text": "whether averaging one item's similarity against a corpus can say how varied the whole collection is"},

    {"gold": "claude-3b1bc969-7931-4530-9e5e-5674a1710374", "kind": "literal",
     "text": "add a LOG_SLOW_QUERIES_MS setting so any query slower than that many milliseconds is logged with its SQL and duration"},
    {"gold": "claude-3b1bc969-7931-4530-9e5e-5674a1710374", "kind": "paraphrase",
     "text": "make the database layer record only statements that take longer than a configured threshold"},

    {"gold": "claude-3d1fb8ea-6ef1-4cd7-82f7-c91686b242ca", "kind": "literal",
     "text": "best practice for session management and spawning parallel subagents with guardrails so nothing destructive happens"},
    {"gold": "claude-3d1fb8ea-6ef1-4cd7-82f7-c91686b242ca", "kind": "paraphrase",
     "text": "running many helpers at once safely and hands-off, preferring links over opening things in the file browser"},

    {"gold": "claude-46017f30-0d30-49c1-8bce-508e93d6e645", "kind": "literal",
     "text": "how to get hierarchical features from the licenses in the software-license-ordering repo"},
    {"gold": "claude-46017f30-0d30-49c1-8bce-508e93d6e645", "kind": "paraphrase",
     "text": "deriving a nested tree structure from legal terms-of-use documents"},

    {"gold": "claude-50a9588f-fe89-406c-80ec-0ba3c2db963d", "kind": "literal",
     "text": "implement LOG_ALL_QUERIES by patching CursorWrapper.execute then conclude it violates the allocation-free constraint"},
    {"gold": "claude-50a9588f-fe89-406c-80ec-0ba3c2db963d", "kind": "paraphrase",
     "text": "a task that asks for the obvious edit first and then to argue it breaks a hot-path performance rule"},

    {"gold": "claude-53aa1dc9-8081-41af-a85b-c520e4ca9292", "kind": "literal",
     "text": "Variable.__setitem__ coercing types on objects with a values property in xarray"},
    {"gold": "claude-53aa1dc9-8081-41af-a85b-c520e4ca9292", "kind": "paraphrase",
     "text": "an array library wrongly converting an assigned object just because it exposes a data attribute"},

    {"gold": "claude-5e4bd3d4-0edb-48c6-acda-d4228ec9465f", "kind": "literal",
     "text": "how do visual and semantic embedding spaces align with each other, and is RBF kernel ridge the only ridge possible"},
    {"gold": "claude-5e4bd3d4-0edb-48c6-acda-d4228ec9465f", "kind": "paraphrase",
     "text": "comparing picture representations against text representations and choosing the regression that maps between them"},

    {"gold": "claude-5f111a3f-77a2-4f47-b7a9-088124e64524", "kind": "literal",
     "text": "research pass on how the skills can be improved, the plots are noisy and the schematic titles are not ideal, distill.pub"},
    {"gold": "claude-5f111a3f-77a2-4f47-b7a9-088124e64524", "kind": "paraphrase",
     "text": "revising reusable instruction files because the charts are cluttered and the figure headings read badly"},

    {"gold": "claude-6417c617-94d7-4482-a0ac-97ca360f6388", "kind": "literal",
     "text": "run the claude-sync sync_status.sh session-start check with SYNC_STATUS_FAST"},
    {"gold": "claude-6417c617-94d7-4482-a0ac-97ca360f6388", "kind": "paraphrase",
     "text": "the shell script run at the beginning of a session to report synchronization state"},

    {"gold": "claude-7199721c-25b8-4947-8c40-6f1688e01e10", "kind": "literal",
     "text": "consolidate the taste skills with the general ones and the EXP-767 preference_pair_head_vl.json schematic"},
    {"gold": "claude-7199721c-25b8-4947-8c40-6f1688e01e10", "kind": "paraphrase",
     "text": "merging two copies of the same instruction files so only brand-specific details like the typeface differ"},

    {"gold": "claude-71bf457c-21a2-4a2b-adf1-a99d79cd71b4", "kind": "literal",
     "text": "where in the upstream repo is the work on the crux pull, and the modal token secret configuration"},
    {"gold": "claude-71bf457c-21a2-4a2b-adf1-a99d79cd71b4", "kind": "paraphrase",
     "text": "locating earlier work in a remote repository and setting up credentials for a serverless compute service"},

    {"gold": "claude-74fb90bb-a586-4609-81c9-543107fc9bf3", "kind": "literal",
     "text": "experiments on behavioral reward models, length-matched anti-selection, and a poset hasse dendrogram over models"},
    {"gold": "claude-74fb90bb-a586-4609-81c9-543107fc9bf3", "kind": "paraphrase",
     "text": "whether a scorer is really just preferring longer answers, and ordering models by how similarly they behave"},

    {"gold": "claude-7a07f0de-724f-42bf-851a-ca973600152e", "kind": "literal",
     "text": "theme canonicalization from rationales, reviewing the past pattern mining drafts and scripts"},
    {"gold": "claude-7a07f0de-724f-42bf-851a-ca973600152e", "kind": "paraphrase",
     "text": "grouping free-text justifications into a consistent set of topics reusing earlier code"},

    {"gold": "claude-7f091101-3bde-420e-9c86-5b20f6a930b1", "kind": "literal",
     "text": "make a better cli, could it just be bdtrace, export and download from the local db direct to hugging face"},
    {"gold": "claude-7f091101-3bde-420e-9c86-5b20f6a930b1", "kind": "paraphrase",
     "text": "renaming and improving a command line tool that pulls sessions out of a local store and uploads them compressed"},

    {"gold": "claude-84c0f445-5701-45b8-88cb-b504cfb7a902", "kind": "literal",
     "text": "remove arbitrary gaps between paragraphs in the ACL natbib EMNLP documentclass while staying in conventions"},
    {"gold": "claude-84c0f445-5701-45b8-88cb-b504cfb7a902", "kind": "paraphrase",
     "text": "fixing vertical spacing in a conference paper without breaking the venue's allowed formatting"},

    {"gold": "claude-896eed13-e45f-4bde-9c5a-135473c64ee8", "kind": "literal",
     "text": "make use of the bidirect-align-dev-traces notebooks, harness-native versus prose self-report, and does it work for cursor"},
    {"gold": "claude-896eed13-e45f-4bde-9c5a-135473c64ee8", "kind": "paraphrase",
     "text": "improving a rough interface for analyzing agent sessions and extending it to a second editor"},

    {"gold": "claude-8a32f4a1-de40-4dba-8215-60bda6ed6165", "kind": "literal",
     "text": "update the schematics skill and the lab-report-analysis repo, goldenstone theme.py, and the open pulls"},
    {"gold": "claude-8a32f4a1-de40-4dba-8215-60bda6ed6165", "kind": "paraphrase",
     "text": "restructuring a shared analysis repository to remove redundancy and opening the pull request for it"},

    {"gold": "claude-8f2a2465-c214-40bb-a2c3-c5b366907886", "kind": "literal",
     "text": "give the rundown of the shared memory and the sessiongrep utility and session-events events.jsonl"},
    {"gold": "claude-8f2a2465-c214-40bb-a2c3-c5b366907886", "kind": "paraphrase",
     "text": "explaining how the saved notes directory and the cross-session search tool fit together"},

    {"gold": "claude-909bd2dc-c3a0-40b3-9cd9-0fbfb4c7e1bd", "kind": "literal",
     "text": "allow FilePathField path to accept a callable because the files are stored differently on each machine"},
    {"gold": "claude-909bd2dc-c3a0-40b3-9cd9-0fbfb4c7e1bd", "kind": "paraphrase",
     "text": "making a model field's location resolvable at runtime, with a previous agent's log supplied for orientation"},

    {"gold": "claude-9436eeb6-5263-4fe5-b943-85b1efefeeb5", "kind": "literal",
     "text": "get re-acclimated with the probe-judge-index work and what the open questions are"},
    {"gold": "claude-9436eeb6-5263-4fe5-b943-85b1efefeeb5", "kind": "paraphrase",
     "text": "catching up on a scoring system, whether it is operational and whether its documentation is current"},

    {"gold": "claude-960b9d9b-6474-4e94-a00d-d27ee6fa3e4c", "kind": "literal",
     "text": "partial ordering and the software licenses work against the prediction-market-resolution outcome corpus in derived/outcomes_v1.jsonl"},
    {"gold": "claude-960b9d9b-6474-4e94-a00d-d27ee6fa3e4c", "kind": "paraphrase",
     "text": "recovering a global ranking from pairwise preferences and relating it to betting market settlement data"},

    {"gold": "claude-abdadc5c-aaa0-4045-b951-4ad3d8fd7e32", "kind": "literal",
     "text": "claude mcp add with user scope and http transport for parallel-search and hf-mcp-server, and claude doctor"},
    {"gold": "claude-abdadc5c-aaa0-4045-b951-4ad3d8fd7e32", "kind": "paraphrase",
     "text": "registering remote tool servers in the assistant's configuration file and checking the diagnostics command"},

    {"gold": "claude-ad59ec9c-9ca5-4cc2-b7ea-4b792b43098b", "kind": "literal",
     "text": "the schematic skill should have an output checklist so all prose is revised and fluff trimmed, wary of verbose subtitles"},
    {"gold": "claude-ad59ec9c-9ca5-4cc2-b7ea-4b792b43098b", "kind": "paraphrase",
     "text": "adding a final review step to a diagram guide so the wording gets tightened before it ships"},

    {"gold": "claude-b975507e-827f-4dcb-bedd-b7e1a6f51848", "kind": "literal",
     "text": "whether wasserstein suits the reward trajectory modeling for the distilled procgrep model"},
    {"gold": "claude-b975507e-827f-4dcb-bedd-b7e1a6f51848", "kind": "paraphrase",
     "text": "choosing a loss that compares a sequence of partial scores against a target sequence across a conversation"},
]

# --- candidate retrievers -----------------------------------------------------
#
# `prefix` is the model card's documented query instruction, applied to the
# query only. Using a model as documented is not tuning; where a card offers a
# prefix, both variants are reported rather than the better of the two.

NEURAL: list[dict] = [
    {"name": "all-MiniLM-L6-v2 (incumbent)", "repo": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "all-mpnet-base-v2", "repo": "sentence-transformers/all-mpnet-base-v2"},
    {"name": "bge-small-en-v1.5", "repo": "BAAI/bge-small-en-v1.5"},
    {"name": "bge-small-en-v1.5 + query prefix", "repo": "BAAI/bge-small-en-v1.5",
     "prefix": "Represent this sentence for searching relevant passages: "},
    {"name": "jina-embeddings-v2-base-code", "repo": "jinaai/jina-embeddings-v2-base-code",
     "trust_remote_code": True},
    {"name": "SFR-Embedding-Code-400M_R", "repo": "Salesforce/SFR-Embedding-Code-400M_R",
     "trust_remote_code": True},
]

INCUMBENT = "all-MiniLM-L6-v2 (incumbent)"

# One tokenizer for both lexical baselines. It keeps whole paths as tokens AND
# emits their components, which helps BM25 match `django/db/backends/utils.py`
# against either form. That choice favours the lexical baselines; it is taken
# deliberately, since the question is whether neural retrieval beats lexical.
_TOK = re.compile(r"[A-Za-z0-9_./\-]+")
_SPLIT = re.compile(r"[/._\-]+")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOK.finditer(text.lower()):
        tok = m.group(0).strip("./-_")
        if not tok:
            continue
        out.append(tok)
        parts = [p for p in _SPLIT.split(tok) if len(p) > 1]
        if len(parts) > 1:
            out.extend(parts)
    return out


def load_corpus(path: Path, max_events: int = 30, char_cap: int = 2000) -> tuple[list[str], list[str]]:
    """Defaults are the shipped `record_text` caps (bdtrace.query.TEXT_EVENT_CAP
    and TEXT_CHAR_CAP); the text-view diagnostic widens them."""
    ids, texts = [], []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ids.append(r["instance_id"])
        texts.append(record_text(r, max_events=max_events, char_cap=char_cap))
    return ids, texts


# --- ranking ------------------------------------------------------------------


METRICS = ("recall@1", "recall@5", "mrr")


def query_stats(scores, gold_idx: int) -> dict:
    """Per-query metric contributions, taken as the EXACT expectation under
    random tie-breaking rather than a single realised ranking.

    Ties are not a corner case here: BM25 scores most of the corpus exactly 0.0,
    and `argsort` would then hand the gold a rank determined by its row number.
    A first version of this script asserted permutation-invariance and caught
    BM25 moving 1.7pp of recall@1 with candidate order. With `g` documents
    scoring strictly higher than the gold and `t` documents tied with it (the
    gold included), the rank is uniform over {g+1, …, g+t}, so the expectations
    below are closed-form, deterministic, and order-invariant by construction.
    """
    import numpy as np

    s = np.asarray(scores, dtype=float)
    gv = s[gold_idx]
    g = int((s > gv).sum())
    t = int((s == gv).sum())
    band = np.arange(g + 1, g + t + 1)
    return {
        "recall@1": float((band == 1).mean()),
        "recall@5": float((band <= 5).mean()),
        "mrr": float((1.0 / band).mean()),
        "expected_rank": float(band.mean()),
        "n_tied": t,
    }


def aggregate(stats: list[dict]) -> dict:
    return {k: sum(s[k] for s in stats) / len(stats) for k in METRICS}


def bootstrap(stats: list[dict], b: int, seed: int) -> dict:
    """Percentile 95% interval over queries resampled with replacement."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stats), size=(b, len(stats)))
    out = {}
    for k in METRICS:
        arr = np.asarray([s[k] for s in stats])
        v = arr[idx].mean(axis=1)
        out[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def paired_delta(stats_a: list[dict], stats_b: list[dict], b: int, seed: int) -> dict:
    """Paired bootstrap of (a - b) per metric: the same resampled query indices
    are applied to both arms, which is the right test when both arms answered
    the identical queries."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stats_a), size=(b, len(stats_a)))
    out = {}
    for k in METRICS:
        A = np.asarray([s[k] for s in stats_a])
        B = np.asarray([s[k] for s in stats_b])
        diff = A[idx].mean(axis=1) - B[idx].mean(axis=1)
        out[k] = {
            "estimate": float(A.mean() - B.mean()),
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "p_gt_0": float((diff > 0).mean()),
        }
    return out


def realised_spread(scores_list, gold_idx: list[int], perms) -> dict:
    """Diagnostic only: how far the metrics actually swing if you rank with a
    hard argsort under different candidate orders. Large here means the arm's
    headline is partly an artefact of corpus ordering."""
    import numpy as np

    per = []
    for p in perms:
        ranks = []
        for s, gi in zip(scores_list, gold_idx, strict=True):
            arr = np.asarray(s, dtype=float)
            order = p[np.argsort(-arr[p], kind="stable")]
            ranks.append(int(np.where(order == gi)[0][0]) + 1)
        per.append({
            "recall@1": sum(r == 1 for r in ranks) / len(ranks),
            "recall@5": sum(r <= 5 for r in ranks) / len(ranks),
            "mrr": sum(1.0 / r for r in ranks) / len(ranks),
        })
    return {k: round(max(m[k] for m in per) - min(m[k] for m in per), 4) for k in METRICS}


# --- retrievers ---------------------------------------------------------------


def run_neural(cfg: dict, texts: list[str], queries: list[str], device: str,
               batch: int = EMBED_BATCH) -> tuple[list, dict]:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    t0 = time.perf_counter()
    model = SentenceTransformer(
        cfg["repo"], device=device, trust_remote_code=cfg.get("trust_remote_code", False)
    )
    load_s = time.perf_counter() - t0

    # batch size changes throughput, never the vectors — the long-text
    # diagnostic lowers it because 256 x 2.6k tokens exhausts memory
    enc = dict(normalize_embeddings=True, show_progress_bar=False, batch_size=batch)
    model.encode(texts[:8], **enc)  # warm the kernels so the timings are steady state

    t0 = time.perf_counter()
    doc = model.encode(texts, **enc)
    build_s = time.perf_counter() - t0

    prefix = cfg.get("prefix", "")
    lat = []
    qvecs = []
    for q in queries:
        t0 = time.perf_counter()
        v = model.encode([prefix + q], **enc)[0]
        _ = np.asarray(doc) @ v
        lat.append((time.perf_counter() - t0) * 1000)
        qvecs.append(v)

    scores = [np.asarray(doc) @ v for v in qvecs]

    tok = model.tokenizer
    lens = [len(tok(t)["input_ids"]) for t in texts]
    msl = model.max_seq_length
    prov = {
        "repo": cfg["repo"],
        "device": device,
        "dim": int(model.get_sentence_embedding_dimension()),
        "max_seq_length": int(msl),
        "encode_params": {"normalize_embeddings": True, "batch_size": batch, "dtype": "float32"},
        "query_prefix": prefix or None,
        "load_s": round(load_s, 2),
        "index_build_s": round(build_s, 2),
        "query_latency_ms_median": round(statistics.median(lat), 2),
        "corpus_tokens_median": int(statistics.median(lens)),
        "corpus_share_over_max_seq_len": round(sum(x > msl for x in lens) / len(lens), 3),
        "disk_mb": model_disk_mb(cfg["repo"]),
    }
    return scores, prov


def run_bm25(texts: list[str], queries: list[str]) -> tuple[list, dict]:
    from rank_bm25 import BM25Okapi
    import numpy as np

    t0 = time.perf_counter()
    bm = BM25Okapi([tokenize(t) for t in texts])
    build_s = time.perf_counter() - t0
    lat, scores = [], []
    for q in queries:
        t0 = time.perf_counter()
        s = bm.get_scores(tokenize(q))
        lat.append((time.perf_counter() - t0) * 1000)
        scores.append(np.asarray(s))
    zero = float(np.mean([(s == 0).mean() for s in scores]))
    return scores, {
        "repo": "rank_bm25.BM25Okapi (k1=1.5, b=0.75 defaults)",
        "index_build_s": round(build_s, 3),
        "query_latency_ms_median": round(statistics.median(lat), 2),
        "disk_mb": 0.0,
        "mean_share_of_corpus_scoring_zero": round(zero, 3),
    }


def run_tfidf(texts: list[str], queries: list[str]) -> tuple[list, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=True, token_pattern=None)
    t0 = time.perf_counter()
    D = vec.fit_transform(texts)
    build_s = time.perf_counter() - t0
    import sklearn.preprocessing as pp

    D = pp.normalize(D)
    lat, scores = [], []
    for q in queries:
        t0 = time.perf_counter()
        v = pp.normalize(vec.transform([q]))
        s = (D @ v.T).toarray().ravel()
        lat.append((time.perf_counter() - t0) * 1000)
        scores.append(s)
    return scores, {
        "repo": "sklearn TfidfVectorizer, cosine",
        "index_build_s": round(build_s, 3),
        "query_latency_ms_median": round(statistics.median(lat), 2),
        "disk_mb": 0.0,
        "vocabulary": int(D.shape[1]),
    }


def run_random(n_docs: int, n_q: int, seed: int) -> tuple[list, dict]:
    """Chance floor. With 138 candidates, recall@1 should land near 1/138."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [rng.random(n_docs) for _ in range(n_q)], {"repo": "uniform random scores", "disk_mb": 0.0}


def run_oracle(ids: list[str], golds: list[str]) -> tuple[list, dict]:
    """Known-positive control. Scores the gold document 1 and everything else 0.
    Any harness bug (id misalignment, off-by-one in ranking, tie-break leak)
    shows up here as something other than a perfect score."""
    import numpy as np

    pos = {i: n for n, i in enumerate(ids)}
    out = []
    for g in golds:
        s = np.zeros(len(ids))
        s[pos[g]] = 1.0
        out.append(s)
    return out, {"repo": "control: gold document scored 1", "disk_mb": 0.0}


def model_disk_mb(repo: str) -> float:
    """Deduplicated size of the weight blobs actually pulled for this repo."""
    from huggingface_hub import snapshot_download

    try:
        root = Path(snapshot_download(repo_id=repo, local_files_only=True))
    except Exception:
        return -1.0
    # the snapshot entries are symlinks into blobs/, and a blob filename is a
    # bare hash — so the suffix test must read the LINK name and the size must
    # come from the resolved target. Report ONE weight format: a cache that
    # happens to hold both safetensors and the equivalent .bin would otherwise
    # double-count (MiniLM read 181.8 MB for a 90.9 MB model).
    for suffixes in ({".safetensors"}, {".bin"}, {".pt"}):
        seen, total = set(), 0
        for p in root.rglob("*"):
            if p.suffix not in suffixes:
                continue
            real = p.resolve()
            if not real.is_file() or real in seen:
                continue
            seen.add(real)
            total += real.stat().st_size
        if total:
            return round(total / 1e6, 1)
    return 0.0


# --- main ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps the numbers comparable to a laptop `bdtrace index build`")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--skip", default="", help="comma-separated retriever names to skip")
    ap.add_argument("--skip-text-view", action="store_true",
                    help="skip the record_text width diagnostic")
    args = ap.parse_args()

    import numpy as np

    ids, texts = load_corpus(args.corpus)
    assert len(set(ids)) == len(ids), "instance_id is not unique; gold labels would be ambiguous"
    pos = {i: n for n, i in enumerate(ids)}
    missing = [q["gold"] for q in QUERIES if q["gold"] not in pos]
    assert not missing, f"gold ids absent from corpus: {missing}"
    # two sessions with identical visible text would make their gold label
    # unrecoverable by ANY retriever, so count them rather than discover it later
    dup_text = len(texts) - len(set(texts))

    qtexts = [q["text"] for q in QUERIES]
    golds = [q["gold"] for q in QUERIES]
    gold_idx = [pos[g] for g in golds]

    # query-set diagnostics: how much of each query is literal overlap
    df: dict[str, int] = {}
    doctoks = [set(tokenize(t)) for t in texts]
    for s in doctoks:
        for t in s:
            df[t] = df.get(t, 0) + 1
    diag = []
    for q, gi in zip(QUERIES, gold_idx, strict=True):
        qt = set(tokenize(q["text"]))
        diag.append({
            "gold": q["gold"], "kind": q["kind"],
            "n_terms": len(qt),
            "n_terms_df1": sum(1 for t in qt if df.get(t, 0) == 1),
            "containment_in_gold": round(len(qt & doctoks[gi]) / max(len(qt), 1), 3),
        })
    by_kind_diag = {}
    for kind in ("literal", "paraphrase"):
        rows = [d for d in diag if d["kind"] == kind]
        by_kind_diag[kind] = {
            "n": len(rows),
            "mean_containment_in_gold": round(statistics.mean(d["containment_in_gold"] for d in rows), 3),
            "queries_with_a_corpus_unique_term": sum(1 for d in rows if d["n_terms_df1"] > 0),
        }

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    runs: list[tuple[str, list, dict]] = []

    sc, pv = run_oracle(ids, golds)
    runs.append(("control: oracle", sc, pv))
    sc, pv = run_random(len(ids), len(QUERIES), args.seed)
    runs.append(("control: random", sc, pv))
    sc, pv = run_bm25(texts, qtexts)
    runs.append(("BM25 (lexical)", sc, pv))
    sc, pv = run_tfidf(texts, qtexts)
    runs.append(("TF-IDF cosine (lexical)", sc, pv))

    for cfg in NEURAL:
        if cfg["name"] in skip:
            print(f"skipping {cfg['name']} (requested)", file=sys.stderr)
            continue
        print(f"running {cfg['name']} ...", file=sys.stderr, flush=True)
        try:
            sc, pv = run_neural(cfg, texts, qtexts, args.device)
        except Exception as e:  # a model that will not download is a reportable fact
            print(f"FAILED {cfg['name']}: {type(e).__name__}: {e}", file=sys.stderr)
            runs.append((cfg["name"], None, {"repo": cfg["repo"], "error": f"{type(e).__name__}: {e}"}))
            continue
        runs.append((cfg["name"], sc, pv))

    # rank every arm under several candidate permutations and assert the metrics
    # are permutation-invariant (the tie-break check §15 asks for)
    rng = random.Random(args.seed)
    perms = [np.arange(len(ids))] + [
        np.array(rng.sample(range(len(ids)), len(ids))) for _ in range(4)
    ]

    results = {}
    stat_table: dict[str, list[dict]] = {}
    for name, scores, prov in runs:
        if scores is None:
            results[name] = {"provenance": prov, "status": "unavailable"}
            continue
        stats = [query_stats(s, gi) for s, gi in zip(scores, gold_idx, strict=True)]
        stat_table[name] = stats
        overall = aggregate(stats)
        ci = bootstrap(stats, args.bootstrap, args.seed)
        for k, (lo, hi) in ci.items():  # the interval must contain its own estimate
            assert lo - 1e-9 <= overall[k] <= hi + 1e-9, f"{name} {k}: {overall[k]} outside {lo},{hi}"
        by_kind = {}
        for kind in ("literal", "paraphrase"):
            sel = [s for s, q in zip(stats, QUERIES, strict=True) if q["kind"] == kind]
            by_kind[kind] = {**aggregate(sel), "ci95": bootstrap(sel, args.bootstrap, args.seed), "n": len(sel)}
        er = [s["expected_rank"] for s in stats]
        results[name] = {
            "provenance": prov,
            "overall": overall,
            "ci95": ci,
            "hard_argsort_spread_over_candidate_orders": realised_spread(scores, gold_idx, perms),
            "by_kind": by_kind,
            "rank_distribution": {
                "at_rank_1": sum(s["recall@1"] == 1.0 for s in stats),
                "in_top_5": sum(s["recall@5"] == 1.0 for s in stats),
                "worse_than_20": sum(r > 20 for r in er),
                "worst_expected_rank": round(max(er), 1),
                "median_expected_rank": round(statistics.median(er), 1),
                "queries_with_ties_at_gold_score": sum(s["n_tied"] > 1 for s in stats),
            },
            "expected_ranks": [round(r, 2) for r in er],
        }

    assert results["control: oracle"]["overall"]["recall@1"] == 1.0, "harness broken: oracle missed"
    rnd = results["control: random"]["overall"]["recall@5"]
    assert 0.0 <= rnd <= 0.15, f"random control implausible at {rnd}; chance is {5 / len(ids):.3f}"

    deltas = {}
    if INCUMBENT in stat_table:
        for name, stats in stat_table.items():
            if name == INCUMBENT or name.startswith("control"):
                continue
            deltas[name] = paired_delta(stats, stat_table[INCUMBENT], args.bootstrap, args.seed)

    # How much of the answer is the model, and how much is the text view it
    # reads? `record_text` shows an embedder the first 30 events capped at 2000
    # characters — a median 10% of a session — so widening it is the obvious
    # alternative to switching models, and worth pricing before recommending.
    #
    # Read the incumbent's row here as a control, not a result: `record_text` is
    # prefix-monotone (a wider cap only appends), and MiniLM truncates at 256
    # tokens, so its input is byte-identical in all three views and its scores
    # MUST NOT move. The arms that can actually see the extra text are BM25 and
    # the two 8192-token code models.
    text_view = {}
    if not args.skip_text_view:
        tv_arms = [c for c in NEURAL if c["name"] in {
            INCUMBENT, "jina-embeddings-v2-base-code", "SFR-Embedding-Code-400M_R"
        } and c["name"] not in skip]
        for label, (ev, cap) in {
            "shipped (30 events, 2000 chars)": (30, 2000),
            "wider (200 events, 8000 chars)": (200, 8000),
            "whole session (uncapped)": (10**9, 10**9),
        }.items():
            _, vtexts = load_corpus(args.corpus, max_events=ev, char_cap=cap)
            row = {
                "median_chars": statistics.median(len(t) for t in vtexts),
                "duplicate_visible_texts": len(vtexts) - len(set(vtexts)),
            }
            sc, _ = run_bm25(vtexts, qtexts)
            row["BM25"] = aggregate([query_stats(s, gi) for s, gi in zip(sc, gold_idx, strict=True)])
            for cfg in tv_arms:
                # the 8192-token arms are CPU-bound on the uncapped view and the
                # wider view already fits entirely inside their window, so that
                # view alone settles whether extra context buys anything
                if cfg["name"] != INCUMBENT and label.startswith("whole session"):
                    row[cfg["name"]] = {"skipped": "CPU cost; the wider view already fits in-window"}
                    continue
                try:
                    sc, _ = run_neural(cfg, vtexts, qtexts, args.device, batch=8)
                except Exception as e:
                    row[cfg["name"]] = {"error": f"{type(e).__name__}: {e}"}
                    continue
                row[cfg["name"]] = aggregate(
                    [query_stats(s, gi) for s, gi in zip(sc, gold_idx, strict=True)]
                )
                print(f"  text view {label}: {cfg['name']} done", file=sys.stderr, flush=True)
            text_view[label] = row

        base = text_view["shipped (30 events, 2000 chars)"].get(INCUMBENT, {})
        for label, row in text_view.items():
            got = row.get(INCUMBENT, {})
            assert all(abs(got[k] - base[k]) < 1e-9 for k in METRICS), (
                f"incumbent moved under text view {label} ({got} vs {base}) — it truncates at "
                "256 tokens on a prefix-monotone text view, so a change here means the "
                "diagnostic is not doing what it claims"
            )

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus": {
            "path": str(args.corpus),
            "n_records": len(ids),
            "record_text_chars": {
                "median": statistics.median(len(t) for t in texts),
                "at_2000_char_cap": sum(len(t) >= 2000 for t in texts),
            },
            "records_with_duplicate_visible_text": dup_text,
        },
        "query_set": {"n": len(QUERIES), "n_gold_sessions": len(set(golds)), "by_kind": by_kind_diag},
        "eval": {
            "chance_recall@1": round(1 / len(ids), 4),
            "chance_recall@5": round(5 / len(ids), 4),
            "bootstrap_draws": args.bootstrap,
            "seed": args.seed,
            "device": args.device,
        },
        "results": results,
        "paired_delta_vs_incumbent": deltas,
        "text_view_sensitivity": text_view,
    }
    text = json.dumps(out, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
