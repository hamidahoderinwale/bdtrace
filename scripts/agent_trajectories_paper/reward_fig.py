import sys; sys.path.insert(0,'.')
import pandas as pd, altair as alt
from scripts.theme import register, BLUE, OLIVE, MAGENTA, GREEN
register()
df=pd.DataFrame([
    {'sel':'proc_score (best-of-n)','rate':38.3,'k':'proc'},
    {'sel':'random agent','rate':32.9,'k':'base'},
    {'sel':'worst proc_score','rate':23.7,'k':'base'},
    {'sel':'best model (Claude-4)','rate':54.0,'k':'ref'},
    {'sel':'oracle','rate':66.7,'k':'ref'},
])
order=['oracle','best model (Claude-4)','proc_score (best-of-n)','random agent','worst proc_score']
ch=alt.Chart(df).mark_bar(size=26).encode(
    x=alt.X('rate:Q',title='Resolve rate',scale=alt.Scale(domain=[0,70]),axis=alt.Axis(domain=False,ticks=False)),
    y=alt.Y('sel:N',sort=order,title=None,axis=alt.Axis(domain=False,ticks=False)),
    color=alt.Color('k:N',legend=None,scale=alt.Scale(domain=['proc','base','ref'],range=[BLUE,OLIVE,GREEN])),
).properties(width=380,height=150,title='Trajectory selection by procedural score')
ch.save('docs/papers/figures/fig_reward_selection.png',scale_factor=2)
print('wrote fig_reward_selection.png')
