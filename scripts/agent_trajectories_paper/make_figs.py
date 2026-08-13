import sys, json
sys.path.insert(0, '.')
import numpy as np, pandas as pd, altair as alt
from collections import Counter
from scipy.spatial.distance import jensenshannon
from scripts.theme import register, BLUE, COPPER, OLIVE, GREEN, MAGENTA
register()
OUT='docs/papers/figures'

rows=[json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
AG_ORDER=['Claude-3','Claude-3.5','Claude-3.7-thinking','Claude-4','GPT-4','GPT-4o','DARS+R1','Agentless+Claude-3.5','Moatless+V3']

# ---------- FIG 1: JSD matrix (per-agent canonical-atom distributions) ----------
alphabet=sorted({a for r in rows for a in r['canonical']})
ai={a:i for i,a in enumerate(alphabet)}
def dist(seqs):
    v=np.zeros(len(alphabet))
    for s in seqs:
        for a in s: v[ai[a]]+=1
    return v/ v.sum() if v.sum() else v
by_agent={}
for r in rows: by_agent.setdefault(r['agent'],[]).append(r['canonical'])
means={a:dist(by_agent[a]) for a in AG_ORDER if a in by_agent}
ags=[a for a in AG_ORDER if a in means]
recs=[]
for a in ags:
    for b in ags:
        j=float(jensenshannon(means[a],means[b],base=2)); j=0.0 if np.isnan(j) else j
        recs.append({'a':a,'b':b,'jsd':j})
dfj=pd.DataFrame(recs)
heat=alt.Chart(dfj).mark_rect().encode(
    x=alt.X('a:N',sort=ags,title=None,axis=alt.Axis(labelAngle=-40,domain=False,ticks=False)),
    y=alt.Y('b:N',sort=ags,title=None,axis=alt.Axis(domain=False,ticks=False)),
    color=alt.Color('jsd:Q',title='JSD',scale=alt.Scale(scheme='teals')),
).properties(width=360,height=360,title='Pairwise procedural divergence by agent')
heat.save(f'{OUT}/fig_jsd_matrix_full_canonical.png',scale_factor=2)
print('wrote fig_jsd_matrix_full_canonical.png  (%d agents)'%len(ags))

# ---------- FIG 2: pass rate by trajectory length ----------
pf=json.load(open('output/paper2_pilot/extended_pass_fail.json'))
resolved_by_sub={k:set(v.get('resolved',[])) for k,v in pf.items()}
# map submission -> resolved set; trajectory resolved if instance in that set
subs=set(r['submission'] for r in rows)
match=sum(1 for s in subs if s in resolved_by_sub)
def is_res(r):
    s=resolved_by_sub.get(r['submission'])
    return (r['instance_id'] in s) if s is not None else None
lab=[(len(r['canonical']), is_res(r)) for r in rows]
lab=[(L,o) for L,o in lab if o is not None]
print('length figure: %d/%d submissions matched pass/fail; %d labeled traces'%(match,len(subs),len(lab)))
bins=[(1,10),(11,20),(21,30),(31,40),(41,60),(61,9999)]
blabel=['1-10','11-20','21-30','31-40','41-60','>60']
br=[]
for (lo,hi),bl in zip(bins,blabel):
    grp=[o for L,o in lab if lo<=L<=hi]
    if grp: br.append({'bin':bl,'pass_rate':100*np.mean(grp),'n':len(grp)})
dfl=pd.DataFrame(br)
line=alt.Chart(dfl).mark_line(point=alt.OverlayMarkDef(size=60,filled=True),strokeWidth=2,color=BLUE).encode(
    x=alt.X('bin:N',sort=blabel,title='Trajectory length (steps)',axis=alt.Axis(domain=False,ticks=False,labelAngle=0)),
    y=alt.Y('pass_rate:Q',title='Pass rate',scale=alt.Scale(domain=[0,100]),axis=alt.Axis(domain=False,ticks=False)),
).properties(width=420,height=220,title='Pass rate by trajectory length')
line.save(f'{OUT}/fig_regression_length.png',scale_factor=2)
print('wrote fig_regression_length.png'); print(dfl.to_string(index=False))

# ---------- FIG 3: BPE vs PrefixSpan V-measure ----------
dfb=pd.DataFrame([
    {'method':'BPE (K=64)','v':0.606},
    {'method':'BPE (K=128)','v':0.626},
    {'method':'PrefixSpan (K=64)','v':0.505},
])
bar=alt.Chart(dfb).mark_bar(size=34).encode(
    x=alt.X('v:Q',title='V-measure vs agent labels',scale=alt.Scale(domain=[0,0.7]),axis=alt.Axis(domain=False,ticks=False)),
    y=alt.Y('method:N',sort=['BPE (K=128)','BPE (K=64)','PrefixSpan (K=64)'],title=None,axis=alt.Axis(domain=False,ticks=False)),
    color=alt.Color('method:N',legend=None,scale=alt.Scale(domain=['BPE (K=64)','BPE (K=128)','PrefixSpan (K=64)'],range=[BLUE,BLUE,COPPER])),
).properties(width=360,height=130,title='Vocabulary induction: BPE vs PrefixSpan')
bar.save(f'{OUT}/fig_bpe_vs_prefixspan.png',scale_factor=2)
print('wrote fig_bpe_vs_prefixspan.png')
