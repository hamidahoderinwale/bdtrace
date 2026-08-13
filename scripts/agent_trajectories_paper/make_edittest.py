import sys, json
sys.path.insert(0,'.')
import numpy as np, pandas as pd, altair as alt
from scripts.theme import register, GREEN, MAGENTA
register()
OUT='docs/papers/figures'
rows=[json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
pf=json.load(open('output/paper2_pilot/extended_pass_fail.json'))
res={k:set(v.get('resolved',[])) for k,v in pf.items()}
def edit_test(seq):
    e=sum(1 for a in seq if a.startswith('EDIT'))
    t=sum(1 for a in seq if a.startswith('RUN'))
    return e,t
recs=[]
for r in rows:
    e,t=edit_test(r['canonical'])
    if e+t==0: continue
    s=res.get(r['submission'])
    if s is None: continue
    recs.append({'ratio':e/(e+t),'outcome':'pass' if r['instance_id'] in s else 'fail'})
df=pd.DataFrame(recs)
print('n=%d  pass=%d fail=%d'%(len(df),(df.outcome=='pass').sum(),(df.outcome=='fail').sum()))
print(df.groupby('outcome')['ratio'].describe()[['mean','50%']])
# density by outcome
chart=alt.Chart(df).transform_density('ratio',groupby=['outcome'],as_=['ratio','density'],extent=[0,1]).mark_area(opacity=0.5).encode(
    x=alt.X('ratio:Q',title='Edit-to-test ratio',axis=alt.Axis(domain=False,ticks=False)),
    y=alt.Y('density:Q',title='Density',axis=alt.Axis(domain=False,ticks=False,labels=False)),
    color=alt.Color('outcome:N',title=None,scale=alt.Scale(domain=['pass','fail'],range=[GREEN,MAGENTA])),
).properties(width=420,height=200,title='Edit-to-test ratio by outcome')
chart.save(f'{OUT}/fig_regression_edit_test.png',scale_factor=2)
print('wrote fig_regression_edit_test.png')
