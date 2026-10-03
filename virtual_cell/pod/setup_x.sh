#!/usr/bin/env bash
# Pod-side setup and smoke test for the D1/D2 exploratory jobs (xmap / xdesign / xmulti).
cd /root/vc/virtual_cell
python - <<'PY'
import re
p = "src/rho_rule.py"; s = open(p).read()
s = re.sub(r"^H5 = \{.*\}$", "H5 = {}  # pod: ess and rpe1 come from the bundled compact npz", s, flags=re.M)
open(p, "w").write(s)
p = "src/run_xjobs.sh"; s = open(p).read().replace("uv run python", "python"); open(p, "w").write(s)
PY
pip install -q anndata scikit-learn > /root/pip2.log 2>&1
grep -n "^H5" src/rho_rule.py
OMP_NUM_THREADS=4 python src/xmap.py --seeds 2 --pairs smoke --ms 50,100 > /root/xsmoke.log 2>&1
echo SMOKE_DONE >> /root/xsmoke.log
