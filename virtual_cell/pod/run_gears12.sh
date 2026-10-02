#!/usr/bin/env bash
# Pod-side: 12 directed pairs in parallel, 10 seeds each, tuned GEARS (one process per pair, own GEARS work dir).
for i in $(seq 120); do grep -q SETUPDONE /root/setup.log && break; sleep 5; done
cd /root/vc/gears
export PYTHONIOENCODING=utf-8
for p in ess:rpe1 rpe1:ess ess:hepg2 hepg2:ess ess:jurkat jurkat:ess rpe1:hepg2 hepg2:rpe1 rpe1:jurkat jurkat:rpe1 hepg2:jurkat jurkat:hepg2; do
  n=${p/:/_}; mkdir -p /root/w/$n; cp -r /root/gfiles/gene2go_all.pkl /root/gfiles/essential_all_data_pert_genes.pkl /root/gfiles/go_essential_all /root/w/$n/
  GEARS_WORK=/root/w/$n python gears_ctx.py --pairs $p --seeds 10 --loss mse --hidden 64 --epochs 60 --step 20 --lr 3e-3 \
    --out g12_$n.json > /root/vc/results/g12_$n.log 2>&1 &
done
wait
echo FIN > /root/gears12_done
