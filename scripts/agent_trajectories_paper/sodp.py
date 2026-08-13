import sys, json, itertools; sys.path.insert(0,'.')
import numpy as np, pandas as pd, altair as alt
from scipy.spatial.distance import jensenshannon
from scripts.theme import register, GREEN, MAGENTA, OLIVE
register()
rows=[json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
pf=json.load(open('output/paper2_pilot/extended_pass_fail.json'))
res={k:set(v.get('resolved',[])) for k,v in pf.items()}
alpha=sorted({a for r in rows for a in r['canonical']}); ai={a:i for i,a in enumerate(alpha)}
def d(seq):
    v=np.zeros(len(alpha))
    for a in seq: v[ai[a]]+=1
    return v/v.sum() if v.sum() else v
# index by instance -> list of (dist, resolved)
inst={}
for r in rows:
    s=res.get(r['submission'])
    if s is None: continue
    inst.setdefault(r['instance_id'],[]).append((d(r['canonical']), r['instance_id'] in s))
recs=[]
for i,lst in inst.items():
    for (da,ra),(db,rb) in itertools.combinations(lst,2):
        j=float(jensenshannon(da,db,base=2)); j=0.0 if np.isnan(j) else j
        cat='both pass' if (ra and rb) else ('both fail' if (not ra and not rb) else 'mixed')
        recs.append({'jsd':j,'cat':cat})
df=pd.DataFrame(recs)
print('pairs:',len(df)); print(df.groupby('cat')['jsd'].agg(['mean','median','count']))
ch=alt.Chart(df).transform_density('jsd',groupby=['cat'],as_=['jsd','density'],extent=[0,1]).mark_area(opacity=0.45).encode(
    x=alt.X('jsd:Q',title='Procedural distance (JSD)',axis=alt.Axis(domain=False,ticks=False)),
    y=alt.Y('density:Q',title='Density',axis=alt.Axis(domain=False,ticks=False,labels=False)),
    color=alt.Color('cat:N',title=None,scale=alt.Scale(domain=['both pass','mixed','both fail'],range=[GREEN,OLIVE,MAGENTA])),
).properties(width=420,height=200,title='Procedural distance by outcome agreement')
ch.save('docs/papers/figures/fig12_tier1b_sodp.png',scale_factor=2)
print('wrote fig12_tier1b_sodp.png')
