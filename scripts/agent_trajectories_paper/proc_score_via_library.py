"""Option (b): score the corpus through the REAL library scorer.

Makes the paper's `from procgrep.reward import load_spec, score` line true:
collapses the corpus's fine-grained canonical atoms to coarse atoms, then scores
each trajectory via procgrep.reward.score against reward_spec.yaml (no `think`
rules). Prints the per-agent table and compares against the inline heuristic so
we can confirm the library scorer is a faithful source of truth before editing
the paper. Run from repo root.
"""
import sys
import json
from collections import defaultdict

# import the procgrep library scorer (the one the paper cites)
sys.path.insert(0, '/Users/hamidaho/learning-from-dev/procgrep/src')
from procgrep.reward import load_spec, score  # noqa: E402

SPEC = load_spec('scripts/agent_trajectories_paper/reward_spec.yaml')

rows = [json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
pf = json.load(open('output/paper2_pilot/extended_pass_fail.json'))
res = {k: set(v.get('resolved', [])) for k, v in pf.items()}


def to_canon(a):
    if a.startswith('EDIT'): return 'edit'
    if a.startswith('CREATE'): return 'create_file'
    if a.startswith('RUN'): return 'run_test'
    if a.startswith(('OPEN', 'NAV')): return 'read_file'
    if a.startswith(('SEARCH', 'FIND')): return 'search_repo'
    if 'SUBMIT' in a: return 'submit'
    return 'other'


agg = defaultdict(lambda: {'n': 0, 'sum': 0.0, 'p': [], 'f': [], 'trig': defaultdict(int)})
for r in rows:
    s = res.get(r['submission'])
    if s is None:
        continue
    atoms = [to_canon(a) for a in r['canonical']]
    result = score(atoms, SPEC)
    sc = result.proc_score
    resolved = r['instance_id'] in s
    a = agg[r['agent']]
    a['n'] += 1; a['sum'] += sc
    (a['p'] if resolved else a['f']).append(sc)
    for name in result.satisfied_phases + result.triggered_bonuses + result.triggered_penalties:
        a['trig'][name] += 1


def m(xs): return sum(xs) / len(xs) if xs else float('nan')


print("=== proc_score via procgrep.reward.score (library scorer) over reward_spec.yaml ===")
hdr = f"{'agent':22}{'n':>5}{'mean':>7}{'pass':>7}{'fail':>7}{'Δ':>7}"
print(hdr); print('-' * len(hdr))
for a in sorted(agg, key=lambda k: -agg[k]['sum'] / agg[k]['n']):
    d = agg[a]
    mean = d['sum'] / d['n']
    pr, fr = m(d['p']), m(d['f'])
    print(f"{a:22}{d['n']:>5}{mean:>7.3f}{pr:>7.3f}{fr:>7.3f}{(pr - fr):>+7.3f}")
alln = sum(d['n'] for d in agg.values())
allp = [x for d in agg.values() for x in d['p']]
allf = [x for d in agg.values() for x in d['f']]
print('-' * len(hdr))
print(f"{'ALL':22}{alln:>5}{(sum(d['sum'] for d in agg.values())/alln):>7.3f}{m(allp):>7.3f}{m(allf):>7.3f}{(m(allp)-m(allf)):>+7.3f}")

print("\n=== trigger rates per agent (library scorer) ===")
COMPS = ['exploration', 'implementation', 'test_verification', 'completion', 'test_driven', 'edit_streak', 'no_search']
for a in sorted(agg):
    d = agg[a]
    tr = ' '.join(f"{c}={100*d['trig'][c]/d['n']:.0f}" for c in COMPS)
    print(f"  {a:22} {tr}")
print("\n(compare to proc_score_breakdown.py inline-heuristic table; expect close, "
      "minor diffs from require_any min_occurrences semantics = per-atom not summed.)")
