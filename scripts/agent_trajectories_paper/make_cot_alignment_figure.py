"""Rebuild cot_alignment_6agents.png — Tufte-minimal, colored by model family.

Forward coverage (what it says -> what it does) and reverse coverage (what it did
-> what it said), per agent, with IQR whiskers. Color = family. Run from repo root.
"""
import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, '.')
import altair as alt
import pandas as pd
from scripts.theme import register, COPPER, BLUE  # noqa: E402

register()

ROOT = Path('.')
SUMMARY = ROOT / 'output/paper2_pilot/cot_action_alignment_embedding_6agents_summary.json'
OUT = ROOT / 'docs/papers/figures/cot_alignment_6agents.png'

FAMILY = {  # name prefix -> family
    'Claude': 'Claude', 'GPT': 'GPT',
}


def family_of(agent):
    for k, v in FAMILY.items():
        if agent.startswith(k):
            return v
    return 'Other'


d = json.load(open(SUMMARY))
rows = []
for agent, s in d.items():
    fam = family_of(agent)
    for measure, mkey, lokey in (
        ('Forward (says→does)', 'median_forward_coverage', 'iqr_forward_coverage'),
        ('Reverse (did→says)', 'median_reverse_coverage', 'iqr_reverse_coverage'),
    ):
        lo, hi = s[lokey]
        rows.append({'agent': agent, 'family': fam, 'measure': measure,
                     'coverage': s[mkey], 'lo': lo, 'hi': hi})

df = pd.DataFrame(rows)
# order agents by family then forward coverage
order = (df[df.measure.str.startswith('Forward')]
         .sort_values(['family', 'coverage'], ascending=[True, False])['agent'].tolist())

fam_scale = alt.Scale(domain=['Claude', 'GPT'], range=[COPPER, BLUE])
base = alt.Chart(df).encode(
    y=alt.Y('agent:N', sort=order, title=None,
            axis=alt.Axis(domain=False, ticks=False)),
)
err = base.mark_rule(opacity=0.45).encode(
    x=alt.X('lo:Q', title='Action-account coverage',
            scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(domain=False, ticks=False)),
    x2='hi:Q',
    color=alt.Color('family:N', scale=fam_scale, legend=alt.Legend(title='Family')),
    detail='measure:N',
)
dots = base.mark_point(filled=True, size=95).encode(
    x=alt.X('coverage:Q', scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('family:N', scale=fam_scale, legend=None),
    shape=alt.Shape('measure:N', title=None,
                    scale=alt.Scale(range=['circle', 'triangle-up'])),
)
chart = (err + dots).properties(
    width=360, height=26 * len(order),
    title='Narration coverage of behavioral accounts by family'
).configure_legend(orient='bottom')

if OUT.exists():
    shutil.copy(OUT, OUT.with_name('cot_alignment_6agents_prev.png'))
chart.save(str(OUT), scale_factor=2)
print(f'wrote {OUT} (backed up old -> cot_alignment_6agents_prev.png)')
print(df.to_string(index=False))
