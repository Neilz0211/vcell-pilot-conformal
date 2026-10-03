"""Tie-aware Spearman(qA, covB_src) over the 14-model ladder, recomputed from results/ladder_metrics.json.
ladder.py originally ranked with argsort (ties broken by order); scipy's spearmanr uses average ranks for ties. The
per-model means are unchanged, so this needs no rerun. Output: results/ladder_spearman.json.
usage: uv run python src/ladder_spearman.py"""
import json
from pathlib import Path

from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[1] / "results"
L = json.loads((R / "ladder_metrics.json").read_text())
out = {}
for d in ("K562->RPE1", "RPE1->K562"):
    v = [x for x in L[d].values() if isinstance(x, dict) and "qA" in x]
    out[d] = float(spearmanr([x["qA"] for x in v], [x["covB_src"] for x in v])[0])
(R / "ladder_spearman.json").write_text(json.dumps(out, indent=1))
print(out)
