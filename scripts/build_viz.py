#!/usr/bin/env python3
"""
Build self-contained HTML visualization of the eval dataset.

Layout:
  Left (flex):   UMAP scatter, full height.
                 Overlay top-right: color toggles, filter dropdowns, legend.
  Right (360px): top half = pass-rate bar chart; bottom half = instance card.

Color modes: fix_type | repo | pass/fail | coverage | quadrant | n_models_solved
Detail card: fix type, coverage score, per-model outcomes, behavioral metrics.

Usage:
    uv run python scripts/build_viz.py
    open notebooks/plots/eval_explorer.html
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap


def _clean_staged_narrative(raw) -> dict:
    """
    Strip embedding vectors and provenance from staged narrative.
    Returns {behavioral, mechanistic, functional} with only human-readable fields.
    """
    if not raw:
        return {}

    # If already a dict, work with it directly
    if isinstance(raw, dict):
        src = raw
    else:
        # Try JSON parse first
        try:
            src = json.loads(raw)
        except Exception:
            src = {}

    # If it's a flat string (Python repr format), fall back to regex stripping
    if not src and isinstance(raw, str):
        cleaned = re.sub(r"'embedding':\s*\[[^\]]*(?:\[[^\]]*\][^\]]*)*\]", "", raw)
        cleaned = re.sub(r"'provenance':\s*\{[^}]*\}", "", cleaned)
        cleaned = re.sub(r"'grounded_in':\s*\{\}", "", cleaned)
        cleaned = re.sub(r",\s*,", ",", cleaned)
        return {"raw": cleaned.strip()}

    out = {}
    STRIP = {"embedding", "provenance", "grounded_in"}

    for key in ("behavioral", "mechanistic", "functional"):
        block = src.get(key, {})
        if isinstance(block, dict):
            out[key] = {k: v for k, v in block.items() if k not in STRIP and isinstance(v, (str, list))}

    return out

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "output" / "datasets" / "swe_bench_lite_resolved"
RESULTS_DIR = ROOT / "output" / "swebench_results"

_MODELS = {
    "lite_20240402_sweagent_gpt4.json": "GPT-4",
    "lite_20240620_sweagent_claude3.5sonnet.json": "Claude 3.5",
    "lite_20240728_sweagent_gpt4o.json": "GPT-4o",
    "lite_20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128.json": "Qwen2.5",
}


def _load_pass_fail(path: Path) -> dict[str, bool]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {k: bool(v) for k, v in data.items()}
    return {r["instance_id"]: bool(r.get("resolved", r.get("pass", False))) for r in data}


def build_dataset() -> list[dict]:
    labels = pd.read_parquet(DS_DIR / "labels.parquet")
    instance_ids = labels["instance_id"].tolist()

    ## Fix type annotations
    ft_map: dict[str, dict] = {}
    ft_path = DS_DIR / "fix_types.json"
    if ft_path.exists():
        with open(ft_path) as f:
            ft_data = json.load(f)
        ft_map = {r["instance_id"]: r for r in ft_data["results"]}

    ## Multi-model pass/fail
    model_pf: dict[str, dict[str, bool]] = {}
    for fname, label in _MODELS.items():
        p = RESULTS_DIR / fname
        if p.exists():
            model_pf[label] = _load_pass_fail(p)

    ## Coverage scores from 2x2 analysis (mean over models per task)
    cov_map: dict[str, float] = {}
    align_map: dict[str, bool] = {}
    q2x2_path = DS_DIR / "task_solution_2x2.parquet"
    if q2x2_path.exists():
        q2 = pd.read_parquet(q2x2_path)
        cov_mean = q2.groupby("instance_id")["coverage"].mean()
        cov_map = cov_mean.to_dict()
        align_mean = q2.groupby("instance_id")["locally_aligned"].mean()
        align_map = {k: v >= 0.5 for k, v in align_mean.items()}

    ## Behavioral metrics — mean over models per task
    traj_path = ROOT / "output" / "trajectories" / "lite_all_models.parquet"
    beh_map: dict[str, dict] = {}
    if traj_path.exists():
        traj = pd.read_parquet(traj_path)
        beh = traj.groupby("instance_id")[["n_steps", "edit_retry_rate"]].mean()
        beh_map = beh.to_dict(orient="index")

    ## Staged narratives
    staged_map: dict[str, str] = {}
    staged_path = ROOT / "output" / "staged_descriptions.json"
    if staged_path.exists():
        with open(staged_path) as f:
            sd = json.load(f)
        staged_map = {r["instance_id"]: r["staged_narrative"] for r in sd["results"]}

    ## UMAP from edits_set_diff distance matrix
    print("  Computing UMAP...")
    mats = np.load(DS_DIR / "matrices.npz")
    D = mats["edits_set_diff"]
    reducer = umap.UMAP(metric="precomputed", n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(D)

    rows = []
    for idx, iid in enumerate(instance_ids):
        ft = ft_map.get(iid, {})
        repo = iid.rsplit("__", 1)[0] if "__" in iid else iid.split(".")[0]
        cov = float(cov_map.get(iid, 0.0))
        high_cov = cov >= 0.2  # ~median from earlier analysis
        locally_aligned = align_map.get(iid, False)

        # 2x2 quadrant label
        if high_cov and locally_aligned:
            quadrant = "tractable + aligned"
        elif high_cov and not locally_aligned:
            quadrant = "tractable + generic"
        elif not high_cov and locally_aligned:
            quadrant = "novel + aligned"
        else:
            quadrant = "novel + generic"

        # Per-model pass/fail
        model_outcomes = {m: bool(pf.get(iid, False)) for m, pf in model_pf.items()}
        n_models_solved = sum(model_outcomes.values())

        b = beh_map.get(iid, {})
        rows.append({
            "idx": idx,
            "instance_id": iid,
            "repo": repo,
            "fix_type": ft.get("fix_type", "unknown"),
            "confidence": ft.get("confidence", ""),
            "summary": ft.get("summary", ""),
            "n_files": ft.get("n_files_changed", 1),
            "net_lines": ft.get("net_lines", 0),
            "passed": bool(model_pf.get("Claude 3.5", {}).get(iid, False)),
            "model_outcomes": model_outcomes,
            "n_models_solved": n_models_solved,
            "coverage": round(cov, 3),
            "quadrant": quadrant,
            "locally_aligned": locally_aligned,
            "n_steps": round(float(b.get("n_steps", 0)), 1),
            "edit_retry_rate": round(float(b.get("edit_retry_rate", 0)), 3),
            "staged_narrative": _clean_staged_narrative(staged_map.get(iid, "")),
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
        })
    return rows


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SWE-bench Lite — Eval Explorer v2</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0d0f17; color: #e0e0e0;
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}

/* ── header ─────────────────────────────────────────── */
header {
  flex: 0 0 auto;
  padding: 11px 20px;
  border-bottom: 1px solid #22253a;
  display: flex; align-items: baseline; gap: 16px;
}
header h1 { font-size: 14px; font-weight: 600; color: #fff; }
header .sub { font-size: 11px; color: #555; }

/* ── body split ──────────────────────────────────────── */
.body { flex: 1 1 0; display: flex; overflow: hidden; }

/* ── left: scatter + overlay ─────────────────────────── */
.scatter-wrap {
  flex: 1 1 0; position: relative; overflow: hidden;
}
#scatter { display: block; width: 100%; height: 100%; }

/* overlay panel — top-right of scatter */
.overlay {
  position: absolute; top: 12px; right: 12px;
  width: 210px;
  background: rgba(13,15,23,0.88);
  border: 1px solid #2a2d3e;
  border-radius: 6px;
  backdrop-filter: blur(6px);
  padding: 10px 12px;
  display: flex; flex-direction: column; gap: 10px;
}
.overlay-section { display: flex; flex-direction: column; gap: 5px; }
.overlay-label {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; color: #555;
}
.btn-row { display: flex; gap: 4px; flex-wrap: wrap; }
.btn {
  background: #1a1d2c; border: 1px solid #30334a; color: #bbb;
  padding: 3px 9px; border-radius: 3px; font-size: 11px; cursor: pointer;
  transition: background 0.12s;
}
.btn:hover { background: #242740; }
.btn.active { background: #3451c7; border-color: #4c6ef5; color: #fff; }
select.ov-select {
  width: 100%; background: #1a1d2c; border: 1px solid #30334a; color: #ccc;
  padding: 3px 6px; border-radius: 3px; font-size: 11px; cursor: pointer;
}
.legend-list {
  display: flex; flex-direction: column; gap: 3px;
  max-height: 200px; overflow-y: auto;
}
.legend-item {
  display: flex; align-items: center; gap: 5px;
  font-size: 11px; color: #aaa; cursor: pointer; padding: 1px 0;
  border-radius: 2px; transition: background 0.1s;
}
.legend-item:hover { background: #1a1d2c; }
.legend-item.dimmed { color: #444; }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.filter-count { font-size: 10px; color: #555; }

/* ── right column ────────────────────────────────────── */
.right-col {
  flex: 0 0 340px; width: 340px;
  display: flex; flex-direction: column;
  border-left: 1px solid #22253a;
  overflow: hidden;
}

/* bar chart panel — fixed height so detail card gets the rest */
.bar-panel {
  flex: 0 0 48%;
  display: flex; flex-direction: column;
  border-bottom: 1px solid #22253a; overflow: hidden;
}
.panel-title {
  flex: 0 0 auto;
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; color: #555; padding: 8px 14px 5px;
}
#bars { flex: 1 1 0; display: block; width: 100%; }

/* detail card */
.detail-wrap {
  flex: 1 1 0;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.detail-panel { flex: 1 1 0; overflow-y: auto; padding: 12px 14px; }
.card-empty { color: #444; font-size: 12px; margin-top: 24px; text-align: center; }
.card h2 { font-size: 12px; font-weight: 600; color: #fff; margin-bottom: 6px;
           word-break: break-all; line-height: 1.4; }
.tag {
  display: inline-block; padding: 1px 7px; border-radius: 3px;
  font-size: 10px; font-weight: 700; margin-right: 4px; margin-bottom: 5px;
}
.tag-pass { background: #1a4731; color: #4ade80; }
.tag-fail { background: #3b1a1a; color: #f87171; }
.tag-type { background: #1c2548; color: #93c5fd; }
.tag-repo { background: #241c3b; color: #c4b5fd; }
.dmeta { font-size: 10px; color: #555; margin-bottom: 8px; }
.dsection { margin-top: 10px; }
.dsection-label {
  font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: #555; margin-bottom: 3px;
}
.dsection-body { font-size: 11px; color: #bbb; line-height: 1.6; }

/* ── modal ───────────────────────────────────────────── */
.modal-backdrop {
  position: fixed; inset: 0; z-index: 100;
  background: rgba(0,0,0,0.72);
  display: flex; align-items: center; justify-content: center;
}
.modal {
  background: #131520; border: 1px solid #2a2d3e; border-radius: 8px;
  padding: 28px 32px; max-width: 520px; width: 90%;
  display: flex; flex-direction: column; gap: 14px;
}
.modal h2 { font-size: 14px; font-weight: 600; color: #fff; }
.modal p  { font-size: 12px; color: #aaa; line-height: 1.7; }
.modal p b { color: #ddd; font-weight: 600; }
.modal hr { border: none; border-top: 1px solid #22253a; }
.modal-close {
  align-self: flex-end;
  background: #3451c7; border: none; color: #fff;
  padding: 5px 18px; border-radius: 4px; font-size: 12px;
  cursor: pointer;
}
.modal-close:hover { background: #4c6ef5; }
</style>
</head>
<body>

<!-- intro modal -->
<div class="modal-backdrop" id="modal-backdrop">
  <div class="modal">
    <h2>SWE-bench Lite — Eval Explorer</h2>
    <p>A <b>structural embedding browser</b> over 300 software engineering tasks. Each point is one instance; <b>proximity = procedural similarity</b> — positions are computed by UMAP over pairwise AST edit-operation distances, not semantic embeddings.</p>
    <hr>
    <p><b>Fix types</b> are derived in two stages: (1) hunk-local AST features are extracted from the diff (node types, control-flow signals, API calls); (2) an LLM classifies the patch into a closed vocabulary using those features as grounding. Labels are structurally anchored, not free-form.</p>
    <hr>
    <p><b>Color</b> by fix type, repo, pass/fail, coverage, quadrant, or models solved. <b>Filter</b> by fix type, repo, or model outcome. <b>Click</b> any point to inspect per-model outcomes, behavioral metrics, and fix summary.</p>
    <button class="modal-close" onclick="document.getElementById('modal-backdrop').style.display='none'">Explore</button>
  </div>
</div>

<header>
  <h1>SWE-bench Lite — Eval Explorer</h1>
  <span class="sub">300 instances · UMAP on edit-operation distances · 4 agent models</span>
</header>

<div class="body">

  <!-- scatter + top-right overlay -->
  <div class="scatter-wrap">
    <canvas id="scatter"></canvas>

    <div class="overlay">

      <div class="overlay-section">
        <div class="overlay-label">Color by</div>
        <div class="btn-row">
          <button class="btn active" onclick="setColor('fix_type',this)">Fix type</button>
          <button class="btn" onclick="setColor('repo',this)">Repo</button>
          <button class="btn" onclick="setColor('passed',this)">Pass/fail</button>
          <button class="btn" onclick="setColor('coverage',this)">Coverage</button>
          <button class="btn" onclick="setColor('quadrant',this)">Quadrant</button>
          <button class="btn" onclick="setColor('n_models',this)">Models solved</button>
        </div>
      </div>

      <div class="overlay-section">
        <div class="overlay-label">Filter</div>
        <select class="ov-select" id="sel-fix" onchange="applyFilter()">
          <option value="">All fix types</option>
        </select>
        <select class="ov-select" id="sel-repo" onchange="applyFilter()">
          <option value="">All repos</option>
        </select>
        <select class="ov-select" id="sel-model" onchange="applyFilter()">
          <option value="">Any model outcome</option>
          <option value="solved_any">Solved by ≥1 model</option>
          <option value="solved_all">Solved by all models</option>
          <option value="unsolved">Unsolved by all</option>
        </select>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span class="filter-count" id="fcount"></span>
          <button class="btn" onclick="resetFilter()" style="padding:2px 7px">Reset</button>
        </div>
      </div>

      <div class="overlay-section" id="legend-section">
        <div class="overlay-label" id="legend-label">Legend</div>
        <div class="legend-list" id="legend-list"></div>
      </div>

    </div>
  </div>

  <!-- right column -->
    <div class="right-col">

      <div class="bar-panel">
        <div class="panel-title">Pass rate by fix type</div>
        <canvas id="bars"></canvas>
      </div>

      <div class="detail-wrap">
        <div class="panel-title">Selected instance</div>
        <div class="detail-panel" id="detail">
          <div class="card-empty">Click any point to inspect</div>
        </div>
      </div>

    </div>
</div>

<script>
const RAW = __DATA__;

const FT_COLORS = {
  logic_fix:         '#60a5fa', exception_handling:'#f472b6',
  api_change:        '#34d399', config_fix:        '#fbbf24',
  guard_clause:      '#a78bfa', type_coercion:     '#fb923c',
  refactor:          '#22d3ee', string_fix:        '#e879f9',
  test_fix:          '#86efac', import_fix:        '#fcd34d',
  async_fix:         '#fdba74', loop_fix:          '#94a3b8',
  other:             '#6b7280', unknown:           '#4b5563',
};
const QUADRANT_COLORS = {
  'tractable + aligned':  '#4ade80',
  'tractable + generic':  '#60a5fa',
  'novel + aligned':      '#fbbf24',
  'novel + generic':      '#f87171',
};
const N_MODELS_COLORS = ['#1e293b','#1e3a5f','#1d4ed8','#0072B2','#00cc99'];
const REPO_COLORS = {};
const repoList = [...new Set(RAW.map(d=>d.repo))].sort();
const palette = ['#60a5fa','#f472b6','#34d399','#fbbf24','#a78bfa',
                 '#fb923c','#22d3ee','#e879f9','#86efac','#fcd34d','#94a3b8','#6b7280'];
repoList.forEach((r,i)=>{ REPO_COLORS[r] = palette[i % palette.length]; });

// Coverage gradient: low=dim red, high=bright blue
function covColor(v) {
  const t = Math.min(1, Math.max(0, v / 0.5));
  const r = Math.round(220*(1-t)+30*t);
  const g = Math.round(50*(1-t)+114*t);
  const b = Math.round(50*(1-t)+178*t);
  return `rgb(${r},${g},${b})`;
}

let colorMode = 'fix_type';
let filteredSet = new Set(RAW.map(d=>d.instance_id));
let selectedId = null;

// Populate selects
const selFix = document.getElementById('sel-fix');
const selRepo = document.getElementById('sel-repo');
const selModel = document.getElementById('sel-model');
[...new Set(RAW.map(d=>d.fix_type))].sort().forEach(v=>{
  const o=document.createElement('option'); o.value=v; o.textContent=v; selFix.appendChild(o);
});
repoList.forEach(v=>{
  const o=document.createElement('option'); o.value=v; o.textContent=v; selRepo.appendChild(o);
});

// ── scatter ──────────────────────────────────────────────
const cv = document.getElementById('scatter');
const ctx = cv.getContext('2d');
let W, H, toX, toY;

function resizeScatter() {
  const el = cv.parentElement;
  W = el.clientWidth; H = el.clientHeight;
  cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
  cv.style.width = W+'px'; cv.style.height = H+'px';
  ctx.scale(devicePixelRatio, devicePixelRatio);
  buildScale(); draw();
}

function buildScale() {
  const pad = 36;
  const xs = RAW.map(d=>d.x), ys = RAW.map(d=>d.y);
  const x0=Math.min(...xs), x1=Math.max(...xs);
  const y0=Math.min(...ys), y1=Math.max(...ys);
  toX = v => pad + (v-x0)/(x1-x0) * (W-2*pad);
  toY = v => H - pad - (v-y0)/(y1-y0) * (H-2*pad);
}

function ptColor(d) {
  if (colorMode==='fix_type')  return FT_COLORS[d.fix_type] || '#6b7280';
  if (colorMode==='repo')      return REPO_COLORS[d.repo]   || '#6b7280';
  if (colorMode==='coverage')  return covColor(d.coverage);
  if (colorMode==='quadrant')  return QUADRANT_COLORS[d.quadrant] || '#6b7280';
  if (colorMode==='n_models')  return N_MODELS_COLORS[d.n_models_solved] || '#6b7280';
  return d.passed ? '#4ade80' : '#f87171';
}

function draw() {
  ctx.clearRect(0,0,W,H);
  for (const d of RAW) {
    const active = filteredSet.has(d.instance_id);
    const sel = d.instance_id === selectedId;
    ctx.globalAlpha = active ? (sel ? 1 : 0.78) : 0.1;
    ctx.beginPath();
    ctx.arc(toX(d.x), toY(d.y), sel ? 6 : 3.5, 0, Math.PI*2);
    ctx.fillStyle = ptColor(d);
    ctx.fill();
    if (sel) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  let best=null, bestD=12;
  for (const d of RAW) {
    if (!filteredSet.has(d.instance_id)) continue;
    const dx=toX(d.x)-mx, dy=toY(d.y)-my;
    const dist=Math.sqrt(dx*dx+dy*dy);
    if (dist<bestD) { bestD=dist; best=d; }
  }
  if (best) { selectedId=best.instance_id; draw(); renderCard(best); renderBars(); }
});

// ── legend (updates with colorMode) ──────────────────────
function buildLegend() {
  const list = document.getElementById('legend-list');
  const label = document.getElementById('legend-label');
  list.innerHTML = '';

  let entries;
  if (colorMode==='fix_type') {
    label.textContent = 'Fix type';
    const counts = {};
    RAW.forEach(d=>{ counts[d.fix_type]=(counts[d.fix_type]||0)+1; });
    entries = Object.entries(FT_COLORS)
      .filter(([k])=>counts[k])
      .sort((a,b)=>(counts[b[0]]||0)-(counts[a[0]]||0));
  } else if (colorMode==='repo') {
    label.textContent = 'Repo';
    entries = Object.entries(REPO_COLORS);
  } else if (colorMode==='quadrant') {
    label.textContent = 'Quadrant';
    entries = Object.entries(QUADRANT_COLORS);
  } else if (colorMode==='n_models') {
    label.textContent = 'Models solved';
    entries = [0,1,2,3,4].map(n=>[`${n} model${n===1?'':'s'}`, N_MODELS_COLORS[n]]);
  } else if (colorMode==='coverage') {
    label.textContent = 'Coverage (low→high)';
    entries = [[' low (0.0)', covColor(0)],[' mid (0.2)', covColor(0.2)],[' high (0.4+)', covColor(0.4)]];
  } else {
    label.textContent = 'Outcome (Claude 3.5)';
    entries = [['pass','#4ade80'],['fail','#f87171']];
  }

  entries.forEach(([key, color]) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    const ft = selFix.value, rp = selRepo.value;
    const isActive = colorMode==='fix_type'
      ? (!ft || ft===key)
      : colorMode==='repo'
        ? (!rp || rp===key)
        : true;
    if (!isActive) item.classList.add('dimmed');
    item.innerHTML = `<div class="legend-dot" style="background:${color}"></div>
                      <span>${key.replace(/_/g,' ')}</span>`;
    item.onclick = () => {
      if (colorMode==='fix_type') { selFix.value = selFix.value===key ? '' : key; applyFilter(); }
      else if (colorMode==='repo') { selRepo.value = selRepo.value===key ? '' : key; applyFilter(); }
    };
    list.appendChild(item);
  });
}

// ── filters ──────────────────────────────────────────────
function applyFilter() {
  const ft = selFix.value, rp = selRepo.value, ml = selModel.value;
  filteredSet = new Set(RAW.filter(d=>{
    if (ft && d.fix_type !== ft) return false;
    if (rp && d.repo !== rp) return false;
    if (ml === 'solved_any'  && d.n_models_solved === 0) return false;
    if (ml === 'solved_all'  && d.n_models_solved < Object.keys(d.model_outcomes).length) return false;
    if (ml === 'unsolved'    && d.n_models_solved > 0) return false;
    return true;
  }).map(d=>d.instance_id));
  const n = filteredSet.size;
  document.getElementById('fcount').textContent = n < RAW.length ? `${n} shown` : '';
  buildLegend(); draw(); renderBars();
}
function resetFilter() {
  selFix.value=''; selRepo.value=''; selModel.value='';
  filteredSet = new Set(RAW.map(d=>d.instance_id));
  document.getElementById('fcount').textContent = '';
  buildLegend(); draw(); renderBars();
}
function setColor(mode, btn) {
  colorMode = mode;
  document.querySelectorAll('.btn-row .btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  buildLegend(); draw();
}

// ── bar chart ─────────────────────────────────────────────
const bv = document.getElementById('bars');
const bctx = bv.getContext('2d');
let BW, BH;

function resizeBars() {
  const panel = bv.closest('.bar-panel');
  const titleH = panel.querySelector('.panel-title').offsetHeight;
  BW = panel.clientWidth;
  BH = panel.clientHeight - titleH;
  if (BW <= 0 || BH <= 0) return;
  bv.width = BW*devicePixelRatio; bv.height = BH*devicePixelRatio;
  bv.style.width = BW+'px'; bv.style.height = BH+'px';
  bctx.scale(devicePixelRatio, devicePixelRatio);
  renderBars();
}

function renderBars() {
  bctx.clearRect(0,0,BW,BH);
  const sub = RAW.filter(d=>filteredSet.has(d.instance_id));
  const byFt = {};
  sub.forEach(d=>{
    if (!byFt[d.fix_type]) byFt[d.fix_type]={n:0,pass:0};
    byFt[d.fix_type].n++;
    if (d.passed) byFt[d.fix_type].pass++;
  });
  const rows = Object.entries(byFt)
    .filter(([,v])=>v.n>=2)
    .sort((a,b)=>b[1].pass/b[1].n - a[1].pass/a[1].n);
  if (!rows.length) return;

  const pad = {l:8,r:8,t:6,b:4};
  const labelW = 108, numW = 44;
  const barW = BW - pad.l - pad.r - labelW - numW;
  const rowH = Math.min(24, (BH-pad.t-pad.b)/rows.length);
  const selFt = RAW.find(d=>d.instance_id===selectedId)?.fix_type;

  bctx.font = `10.5px -apple-system,sans-serif`;
  rows.forEach(([ft,{n,pass}], i) => {
    const rate = pass/n;
    const y = pad.t + i*rowH;
    const isSel = ft === selFt;

    bctx.fillStyle = isSel ? '#fff' : (FT_COLORS[ft]||'#888');
    bctx.fillText(ft.replace(/_/g,' '), pad.l, y + rowH*0.7);

    bctx.fillStyle = '#1a1d2a';
    bctx.fillRect(pad.l+labelW, y+3, barW, rowH-6);

    bctx.globalAlpha = isSel ? 1 : 0.65;
    bctx.fillStyle = FT_COLORS[ft]||'#888';
    bctx.fillRect(pad.l+labelW, y+3, barW*rate, rowH-6);
    bctx.globalAlpha = 1;

    bctx.fillStyle = '#666';
    bctx.fillText(`${Math.round(rate*100)}% (${n})`, pad.l+labelW+barW+5, y+rowH*0.7);
  });
}

// ── detail card ───────────────────────────────────────────
function renderCard(d) {
  const fTag = `<span class="tag tag-type">${d.fix_type.replace(/_/g,' ')}</span>`;
  const rTag = `<span class="tag tag-repo">${d.repo}</span>`;

  // Per-model outcomes — full name on hover, short name visible
  const modelTags = Object.entries(d.model_outcomes||{}).map(([m, solved])=>
    `<span class="tag ${solved?'tag-pass':'tag-fail'}" title="${m}">${m}</span>`
  ).join('');

  // Coverage bar
  const covPct = Math.round(d.coverage*100);
  const covBar = `<div style="margin:6px 0 2px">
    <div style="font-size:9px;color:#555;margin-bottom:3px">structural coverage: ${covPct}% · ${d.quadrant}</div>
    <div style="height:5px;background:#1a1d2a;border-radius:2px">
      <div style="height:5px;width:${covPct}%;background:${covColor(d.coverage)};border-radius:2px"></div>
    </div>
  </div>`;

  const behInfo = d.n_steps > 0
    ? `<div class="dmeta">avg ${d.n_steps} steps · retry rate ${(d.edit_retry_rate*100).toFixed(0)}%</div>`
    : '';

  const summary = d.summary
    ? `<div class="dsection"><div class="dsection-label">Fix summary</div>
       <div class="dsection-body">${d.summary}</div></div>` : '';

  // Staged narrative — render structured fields cleanly
  let narrativeHtml = '';
  const sn = d.staged_narrative;
  if (sn && typeof sn === 'object' && !sn.raw) {
    const sections = [];
    if (sn.behavioral) {
      const b = sn.behavioral;
      const lines = [b.claim, b.before && `Before: ${b.before}`, b.after && `After: ${b.after}`].filter(Boolean);
      if (lines.length) sections.push(`<div class="dsection-label">Behavioral</div><div class="dsection-body">${lines.join('<br>')}</div>`);
    }
    if (sn.mechanistic) {
      const m = sn.mechanistic;
      const text = m.mechanism || (Array.isArray(m.steps) ? m.steps.slice(0,3).join(' → ') : '');
      if (text) sections.push(`<div class="dsection-label">Mechanistic</div><div class="dsection-body">${text}</div>`);
    }
    if (sn.functional) {
      const f = sn.functional;
      const text = [f.role, f.system_impact].filter(Boolean).join(' ');
      if (text) sections.push(`<div class="dsection-label">Functional</div><div class="dsection-body">${text}</div>`);
    }
    if (sections.length) narrativeHtml = `<div class="dsection">${sections.join('<div style="height:6px"></div>')}</div>`;
  } else if (sn && sn.raw) {
    narrativeHtml = `<div class="dsection"><div class="dsection-label">Staged narrative</div><div class="dsection-body">${sn.raw}</div></div>`;
  }

  // Nav index
  const idx = RAW.findIndex(r => r.instance_id === d.instance_id);
  const navHtml = `<div style="font-size:10px;color:#444;margin-bottom:6px">${idx+1} / ${RAW.length} &nbsp;
    <span style="cursor:pointer;color:#555" onclick="navigateCard(-1)">◀</span>
    <span style="cursor:pointer;color:#555;margin-left:6px" onclick="navigateCard(1)">▶</span>
  </div>`;

  document.getElementById('detail').innerHTML = `
    ${navHtml}
    <h2>${d.instance_id}</h2>
    <div style="margin-bottom:4px">${modelTags}</div>
    <div>${fTag}${rTag}</div>
    <div class="dmeta">${d.n_files} file(s) · ${d.net_lines>=0?'+':''}${d.net_lines} lines · confidence: ${d.confidence}</div>
    ${covBar}${behInfo}${summary}${narrativeHtml}`;
}

function navigateCard(dir) {
  const ids = RAW.filter(d => filteredSet.has(d.instance_id)).map(d => d.instance_id);
  if (!ids.length) return;
  const cur = ids.indexOf(selectedId);
  const next = ids[(cur + dir + ids.length) % ids.length];
  const d = RAW.find(r => r.instance_id === next);
  if (d) { selectedId = d.instance_id; draw(); renderCard(d); renderBars(); }
}

// ── init ─────────────────────────────────────────────────
function initCanvases() {
  resizeScatter();
  resizeBars();
}

const ro = new ResizeObserver(()=>{ resizeScatter(); resizeBars(); });
ro.observe(document.querySelector('.scatter-wrap'));
ro.observe(document.querySelector('.bar-panel'));
buildLegend();
// Keyboard navigation
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); navigateCard(1); }
  if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   { e.preventDefault(); navigateCard(-1); }
});
// Defer init to ensure layout is settled
requestAnimationFrame(()=> requestAnimationFrame(initCanvases));
</script>
</body>
</html>
"""


def main() -> None:
    print("Building dataset...")
    rows = build_dataset()
    print(f"  {len(rows)} instances")

    html = HTML.replace("__DATA__", json.dumps(rows))
    out = ROOT / "notebooks" / "plots" / "eval_explorer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
