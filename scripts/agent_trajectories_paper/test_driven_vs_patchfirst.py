"""③′ Test-driven vs patch-first procedures.

Reframe of the (ungroundable) reward-hacking run — see REFERENCES.md ③/③′.
Uses the canonical action sequences (trusted) instead of the contaminated patches.

Self-verifying ("test-driven") trajectory := a `run_test` action occurs AFTER the
first `edit` (the agent ran tests to check its own fix). Otherwise "patch-first".

Non-circular claim: does this *behavior* predict resolution? (validates the
`proc_score` edit->run_test milestone, rather than scoring proc_score on the
behavior that defines it). Run from repo root.
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


def proc_score(seq):
    """The reward_verifier.py heuristic, verbatim, so (a) is measured on the real reward."""
    s = [to_canon(a) for a in seq]
    score = 0.0
    fe = next((i for i, a in enumerate(s) if a == 'edit'), None)
    pre = s[:fe] if fe is not None else s
    if sum(1 for a in pre if a in ('search_repo', 'read_file')) >= 2: score += 0.10
    if 'edit' in s: score += 0.15
    for i, a in enumerate(s):
        if a == 'edit' and 'run_test' in s[i + 1:i + 6]: score += 0.25; break
    if 'submit' in s: score += 0.10
    if fe is not None and 'run_test' in s[:fe]: score += 0.10
    run = 0
    for a in s:
        run = run + 1 if a == 'edit' else 0
        if run >= 5: score -= 0.15; break
    if fe is not None and not any(a in ('search_repo', 'read_file') for a in pre): score -= 0.05
    return max(0.0, min(1.0, score))


def self_verifying(seq):
    s = [to_canon(a) for a in seq]
    fe = next((i for i, a in enumerate(s) if a == 'edit'), None)
    if fe is None:
        return None  # never edited — exclude (can't be test-driven about a fix)
    return 'run_test' in s[fe + 1:]


def rate(pairs):
    n = len(pairs)
    return (100 * sum(r for _, r in pairs) / n, n) if n else (float('nan'), 0)


buckets = {True: [], False: []}
per_agent = defaultdict(lambda: {True: [], False: []})
ps_by_group = {True: [], False: []}       # (a) proc_score for test-driven vs patch-first
ps_by_outcome = {True: [], False: []}     # does proc_score track resolution at all?
for r in rows:
    s = res.get(r['submission'])
    if s is None:
        continue
    sv = self_verifying(r['canonical'])
    if sv is None:
        continue
    resolved = r['instance_id'] in s
    buckets[sv].append((r['instance_id'], resolved))
    per_agent[r['agent']][sv].append((r['instance_id'], resolved))
    ps = proc_score(r['canonical'])
    ps_by_group[sv].append(ps)
    ps_by_outcome[resolved].append(ps)

td_r, td_n = rate(buckets[True])
pf_r, pf_n = rate(buckets[False])
print("=== Test-driven vs patch-first (resolution rate) ===")
print(f"  test-driven (run_test after first edit): {td_r:5.1f}%   n={td_n}")
print(f"  patch-first (no test after first edit)  : {pf_r:5.1f}%   n={pf_n}")
print(f"  delta                                   : {td_r - pf_r:+5.1f} pp")

print("\n=== Per-agent (resolution: test-driven / patch-first | n_td,n_pf) ===")
for a in sorted(per_agent):
    tr, tn = rate(per_agent[a][True])
    pr, pn = rate(per_agent[a][False])
    if tn + pn < 20:
        continue
    print(f"  {a:30} {tr:5.1f}% / {pr:5.1f}%   (n={tn},{pn})")


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


print("\n=== (a) Does proc_score SEPARATE test-driven vs patch-first? ===")
print(f"  mean proc_score  test-driven: {mean(ps_by_group[True]):.3f}   patch-first: {mean(ps_by_group[False]):.3f}")
print("  (separation is ~tautological: proc_score credits edit->run_test by construction)")

print("\n=== Does proc_score track RESOLUTION at all? (the real validation) ===")
print(f"  mean proc_score  resolved: {mean(ps_by_outcome[True]):.3f}   unresolved: {mean(ps_by_outcome[False]):.3f}")
print(f"  delta: {mean(ps_by_outcome[True]) - mean(ps_by_outcome[False]):+.3f}")
