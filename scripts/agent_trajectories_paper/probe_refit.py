import json, numpy as np
from collections import Counter, defaultdict
rows=[json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
AG=['Claude-3','Claude-3.5','Claude-3.7-thinking','Claude-4','GPT-4','GPT-4o','DARS+R1','Agentless+Claude-3.5','Moatless+V3']
def stage(a):
    if a.startswith(('SEARCH','FIND','OPEN','NAV')): return 'browse'
    if a.startswith(('EDIT','CREATE')): return 'edit'
    if a.startswith('RUN'): return 'test'
    if 'SUBMIT' in a: return 'finish'
    return 'other'

def probe(seqs, mapfn):
    # first-order Markov next-token prediction, 5-fold by trajectory
    seqs=[[mapfn(a) for a in s] for s in seqs]
    n=len(seqs); idx=np.arange(n); rng=np.random.RandomState(0); rng.shuffle(idx)
    folds=np.array_split(idx,5)
    accs=[]; bases=[]
    for f in range(5):
        test=set(folds[f].tolist()); train=[seqs[i] for i in range(n) if i not in test]
        trans=defaultdict(Counter); marg=Counter()
        for s in train:
            for a,b in zip(s,s[1:]): trans[a][b]+=1; marg[b]+=1
        pred={a:c.most_common(1)[0][0] for a,c in trans.items()}
        base_tok=marg.most_common(1)[0][0] if marg else None
        hit=tot=bhit=0
        for i in range(n):
            if i not in test: continue
            s=seqs[i]
            for a,b in zip(s,s[1:]):
                tot+=1
                if pred.get(a)==b: hit+=1
                if base_tok==b: bhit+=1
        if tot: accs.append(hit/tot); bases.append(bhit/tot)
    return 100*np.mean(bases), 100*np.mean(accs)

by={a:[r['canonical'] for r in rows if r['agent']==a] for a in AG}
print(f"{'Agent':22} | next-action base/acc/Δ | next-stage base/acc/Δ")
for a in AG:
    s=by[a]
    nb,na=probe(s, lambda x:x)
    sb,sa=probe(s, stage)
    print(f"{a:22} |  {nb:4.0f} {na:4.0f} {na-nb:+4.0f}      |  {sb:4.0f} {sa:4.0f} {sa-sb:+4.0f}")
