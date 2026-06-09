"""Procedural-score breakdown across all agents.

Decomposes the proc_score heuristic (as actually run in reward_verifier.py) into
its components and reports, per agent: n, mean score, pass/fail score split, and
the trigger rate of each component. Run from repo root.

NOTE: this scores the *inline Python heuristic* that the runs used — NOT the
paper's YAML spec (which references a `think` action absent from canonicalization
and so cannot be evaluated here). See REFERENCES.md / the spec-mismatch note.
"""
import json
from collections import defaultdict

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


# component name -> (signed weight, trigger function on canonical list s)
def first_edit(s): return next((i for i, a in enumerate(s) if a == 'edit'), None)


def comp_triggers(s):
    fe = first_edit(s)
    pre = s[:fe] if fe is not None else s
    t = {}
    t['exploration(+.10)'] = sum(1 for a in pre if a in ('search_repo', 'read_file')) >= 2
    t['implementation(+.15)'] = 'edit' in s
    t['test_verif(+.25)'] = any(a == 'edit' and 'run_test' in s[i + 1:i + 6] for i, a in enumerate(s))
    t['completion(+.10)'] = 'submit' in s
    t['test_driven(+.10)'] = fe is not None and 'run_test' in s[:fe]
    streak = run = False
    r = 0
    for a in s:
        r = r + 1 if a == 'edit' else 0
        if r >= 5: streak = True; break
    t['edit_streak(-.15)'] = streak
    t['no_search(-.05)'] = fe is not None and not any(a in ('search_repo', 'read_file') for a in pre)
    return t


W = {'exploration(+.10)': .10, 'implementation(+.15)': .15, 'test_verif(+.25)': .25,
     'completion(+.10)': .10, 'test_driven(+.10)': .10, 'edit_streak(-.15)': -.15, 'no_search(-.05)': -.05}
COMPS = list(W)


def score_of(trig):
    return max(0.0, min(1.0, sum(W[c] for c in COMPS if trig[c])))


agg = defaultdict(lambda: {'n': 0, 'sum': 0.0, 'p': [], 'f': [], 'trig': defaultdict(int)})
for r in rows:
    s = res.get(r['submission'])
    if s is None:
        continue
    cs = [to_canon(a) for a in r['canonical']]
    trig = comp_triggers(cs)
    sc = score_of(trig)
    resolved = r['instance_id'] in s
    a = agg[r['agent']]
    a['n'] += 1; a['sum'] += sc
    (a['p'] if resolved else a['f']).append(sc)
    for c in COMPS:
        if trig[c]: a['trig'][c] += 1


def m(xs): return sum(xs) / len(xs) if xs else float('nan')


hdr = f"{'agent':22}{'n':>5}{'mean':>7}{'pass':>7}{'fail':>7}{'Δ':>7}  | trigger rates"
print(hdr); print('-' * len(hdr))
for a in sorted(agg, key=lambda k: -agg[k]['sum'] / agg[k]['n']):
    d = agg[a]
    mean = d['sum'] / d['n']
    pr, fr = m(d['p']), m(d['f'])
    tr = ' '.join(f"{c.split('(')[0]}={100*d['trig'][c]/d['n']:.0f}" for c in COMPS)
    print(f"{a:22}{d['n']:>5}{mean:>7.3f}{pr:>7.3f}{fr:>7.3f}{(pr-fr):>+7.3f}  | {tr}")

# corpus totals
allp = [x for d in agg.values() for x in d['p']]
allf = [x for d in agg.values() for x in d['f']]
alln = sum(d['n'] for d in agg.values())
print('-' * len(hdr))
print(f"{'ALL':22}{alln:>5}{(sum(d['sum'] for d in agg.values())/alln):>7.3f}{m(allp):>7.3f}{m(allf):>7.3f}{(m(allp)-m(allf)):>+7.3f}")
