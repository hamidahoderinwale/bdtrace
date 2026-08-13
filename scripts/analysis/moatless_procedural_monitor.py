"""
Tree-node procedural monitor for Moatless MCTS trajectories.

Applies to the MCTS tree structure: after each node expansion,
checks whether the current path from root matches failure-associated patterns.
Reports which paths would have triggered intervention and at what depth.

This is the offline version — it reads existing Moatless trajectories and
shows where monitoring would have fired. The online version would plug into
Moatless's node scoring function.

Key findings this addresses:
  - Moatless+DeepSeek never runs tests (0% test rate)
  - 40% stuck-edit rate (5+ consecutive EDIT nodes on the path)
  - 54.7% of all actions are EDIT

Monitoring signal: when the partial path from root to current node
shows >50% EDIT concentration with 0 TEST steps, flag for intervention.
"""
import json, numpy as np
from pathlib import Path
from collections import Counter

CACHE = Path('output/trajectories/.cache/20250111_moatless_deepseek_v3')

ACTION_MAP = {
    'FindFunction':'LOCALIZE','FindClass':'LOCALIZE','FindCode':'LOCALIZE',
    'FindCodeSnippet':'LOCALIZE','SemanticSearch':'SEARCH',
    'ViewCode':'OPEN','RequestCodeChange':'EDIT','StringReplace':'EDIT',
    'RunTests':'TEST','Finish':'SUBMIT',
}

def get_action(node):
    steps = node.get('action_steps',[])
    if not steps: return None
    cls = steps[0].get('action',{}).get('action_args_class','')
    name = cls.split('.')[-1].replace('Args','')
    return ACTION_MAP.get(name)

def get_thought(node):
    steps = node.get('action_steps',[])
    if not steps: return ''
    return steps[0].get('action',{}).get('thoughts','') or ''

def walk_path(node, path=None, depth=0):
    """Walk the tree, yielding (depth, action, thought, partial_path, node)."""
    if path is None: path = []
    act = get_action(node)
    thought = get_thought(node)
    current_path = path + ([act] if act else [])
    if act:
        yield depth, act, thought, current_path, node
    for child in node.get('children',[]):
        yield from walk_path(child, current_path, depth+1)

def monitor_path(partial_path, min_length=5):
    """
    Returns a signal dict if the partial path warrants intervention.
    
    Two signals:
    1. edit_flood: >60% of last N actions are EDIT with no TEST
    2. stuck_edit: 4+ consecutive EDITs
    """
    if len(partial_path) < min_length:
        return None
    signals = []
    # Signal 1: edit flood in recent window
    window = partial_path[-8:]
    edit_frac = window.count('EDIT') / len(window)
    has_test  = 'TEST' in partial_path
    if edit_frac > 0.6 and not has_test:
        signals.append(f'edit_flood ({edit_frac:.0%} edits, no tests)')
    # Signal 2: consecutive edits
    streak = 0
    for a in reversed(partial_path):
        if a == 'EDIT': streak += 1
        else: break
    if streak >= 4:
        signals.append(f'stuck_edit (streak={streak})')
    return signals if signals else None

def analyze_trajectory(filepath):
    d = json.loads(filepath.read_text())
    root = d.get('content',{}).get('root',{})
    iid  = d.get('instance_id','?')
    
    interventions = []
    full_path = []
    for depth, act, thought, partial_path, node in walk_path(root):
        full_path = partial_path
        signals = monitor_path(partial_path)
        if signals:
            interventions.append({
                'depth': depth,
                'action': act,
                'signals': signals,
                'path_so_far': partial_path.copy(),
                'thought_excerpt': thought[:100],
            })
    
    return {
        'instance_id': iid,
        'full_path': full_path,
        'n_nodes': len(full_path),
        'edit_pct': full_path.count('EDIT')/len(full_path) if full_path else 0,
        'test_pct': full_path.count('TEST')/len(full_path) if full_path else 0,
        'first_intervention_depth': interventions[0]['depth'] if interventions else None,
        'n_interventions': len(interventions),
        'would_intervene': bool(interventions),
        'interventions': interventions[:3],  # first 3 only
    }

if __name__ == '__main__':
    files = sorted(CACHE.glob('*.json'))[:100]
    results = [analyze_trajectory(f) for f in files]
    
    import pandas as pd
    df = pd.DataFrame(results)
    print(f'=== Moatless Procedural Monitor ===')
    print(f'Trajectories analyzed: {len(df)}')
    print(f'Would intervene: {df.would_intervene.mean():.1%}')
    print(f'Mean first intervention depth: {df.first_intervention_depth.dropna().mean():.1f}')
    print(f'Mean edit%: {df.edit_pct.mean():.1%}')
    print(f'Mean test%: {df.test_pct.mean():.1%}')
    print()
    print('Sample intervention:')
    first = df[df.would_intervene].iloc[0]
    print(f'  Instance: {first.instance_id}')
    print(f'  Path: {first.full_path}')
    print(f'  First signal at depth {first.first_intervention_depth}:')
    if first.interventions:
        iv = first.interventions[0]
        print(f'    Signals: {iv["signals"]}')
        print(f'    Path so far: {iv["path_so_far"]}')
        print(f'    Thought: {iv["thought_excerpt"]!r}')
    
    # Save
    import json as _json
    out = Path('output/paper2_pilot/moatless_monitor_results.json')
    out.write_text(_json.dumps(results, indent=2))
    print(f'\nSaved → {out}')
