# child_traj/ moved to HuggingFace

The 499 SWE-agent-LM-32B rollout trajectories (`child_traj/*.traj`, 1.5 GB) now live at
the private dataset [`midah/bidirect-distillation-traces`](https://huggingface.co/datasets/midah/bidirect-distillation-traces),
uploaded and count-verified 2026-08-13. Last in-tree at commit `5dc8739`.

Fetch:

```python
from huggingface_hub import snapshot_download
snapshot_download("midah/bidirect-distillation-traces", repo_type="dataset", local_dir="distillation_run/child_traj")
```

`preds.json` and the exit-status YAML were kept in-tree (moved up to this directory);
derived fingerprints (`fingerprints_{parent,child}.jsonl`) were always in-tree and are unchanged.
