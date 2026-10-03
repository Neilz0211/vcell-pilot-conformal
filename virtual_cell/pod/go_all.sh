#!/usr/bin/env bash
# Pod: after a stuck extract (all shards done), run compact + xline, and the D1/D2 jobs in parallel.
pkill -f tahoe_extract.py; pkill -f /root/run.sh; sleep 2
cd /root/vc/virtual_cell
python - <<'PY'
import re
p = "src/rho_rule.py"; s = open(p).read()
s = re.sub(r"^H5 = \{.*\}$", "H5 = {}  # pod: ess and rpe1 come from the bundled compact npz", s, flags=re.M); open(p, "w").write(s)
p = "src/run_xjobs.sh"; s = open(p).read().replace("uv run python", "python"); open(p, "w").write(s)
PY
grep -q "^H5 = {}" src/rho_rule.py && echo H5_PATCHED
mkdir -p results/xlogs
( cd /root/code && python tahoe_compact.py /root/tahoe > /root/compact.log 2>&1 && echo COMPACT_OK >> /root/compact.log \
  && python tahoe_xline.py /root/tahoe --seeds 5 --workers 6 --out /root/tahoe_xline.json > /root/xline.log 2>&1 && echo XLINE_OK >> /root/xline.log ) &
( cd /root/vc/virtual_cell && bash src/run_xjobs.sh 5 > results/xlogs/runner.log 2>&1 ) &
wait
echo ALL_DONE
