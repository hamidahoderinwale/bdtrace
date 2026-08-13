import sys, json, random
import numpy as np
rows=[json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
pf=json.load(open('output/paper2_pilot/extended_pass_fail.json'))
res={k:set(v.get('resolved',[])) for k,v in pf.items()}

def to_canon(a):
    if a.startswith('EDIT'): return 'edit'
    if a.startswith('CREATE'): return 'create_file'
    if a.startswith('RUN'): return 'run_test'          # RUN_PYTEST/RUN_PYTHON_TEST/REPRO etc.
    if a.startswith(('OPEN','NAV')): return 'read_file'
    if a.startswith(('SEARCH','FIND')): return 'search_repo'
    if 'SUBMIT' in a: return 'submit'
    return 'other'

def proc_score(seq):
    s=[to_canon(a) for a in seq]
    score=0.0
    first_edit=next((i for i,a in enumerate(s) if a=='edit'), None)
    pre = s[:first_edit] if first_edit is not None else s
    # exploration: >=2 search/read before first edit
    if sum(1 for a in pre if a in('search_repo','read_file'))>=2: score+=0.10
    # implementation
    if 'edit' in s: score+=0.15
    # test_verification: edit followed by run_test within 5
    for i,a in enumerate(s):
        if a=='edit' and 'run_test' in s[i+1:i+6]: score+=0.25; break
    # structured completion
    if 'submit' in s: score+=0.10
    # bonus test_driven: run_test before first edit
    if first_edit is not None and 'run_test' in s[:first_edit]: score+=0.10
    # penalty edit_streak >=5 consecutive
    run=0
    for a in s:
        run = run+1 if a=='edit' else 0
        if run>=5: score-=0.15; break
    # penalty no search/read before edit
    if first_edit is not None and not any(a in('search_repo','read_file') for a in pre): score-=0.05
    return max(0.0,min(1.0,score))

# build per-instance: {instance: {agent: (proc_score, resolved)}}
inst={}
for r in rows:
    s=res.get(r['submission'])
    if s is None: continue
    resolved = r['instance_id'] in s
    inst.setdefault(r['instance_id'],{})[r['agent']]=(proc_score(r['canonical']), resolved)

insts=[i for i in inst if len(inst[i])>=3]
print(f"instances with >=3 agents: {len(insts)}")

def rate(selector):
    hits=sum(1 for i in insts if selector(inst[i])) ; return 100*hits/len(insts)

# selectors
def best_proc(d): a=max(d,key=lambda k:d[k][0]); return d[a][1]
def worst_proc(d): a=min(d,key=lambda k:d[k][0]); return d[a][1]
def oracle(d): return any(v[1] for v in d.values())
# random: expected resolve over uniform agent pick
rng=random.Random(0)
def rand_rate():
    tot=0
    for i in insts:
        vals=list(inst[i].values())
        tot+=100*np.mean([1 if v[1] else 0 for v in vals])
    return tot/len(insts)
# best-overall agent (global resolve rate)
from collections import Counter,defaultdict
ag_res=defaultdict(lambda:[0,0])
for r in rows:
    s=res.get(r['submission']);
    if s is None: continue
    ag_res[r['agent']][0]+= (r['instance_id'] in s); ag_res[r['agent']][1]+=1
best_agent=max(ag_res,key=lambda a:ag_res[a][0]/ag_res[a][1])
def bestagent(d): return d.get(best_agent,(0,False))[1] if best_agent in d else False

print(f"\nSelection resolve-rate over {len(insts)} instances:")
print(f"  proc_score (best-of-n)   : {rate(best_proc):.1f}%")
print(f"  random agent             : {rand_rate():.1f}%")
print(f"  worst proc_score         : {rate(worst_proc):.1f}%")
print(f"  best-overall agent ({best_agent[:18]}): {rate(bestagent):.1f}%")
print(f"  oracle (any resolves)    : {rate(oracle):.1f}%")
