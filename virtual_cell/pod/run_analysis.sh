#!/usr/bin/env bash
# Pod-side: wait for pseudobulk + public data, then run offset-pair analysis on the new cell lines.
for i in $(seq 900); do test -f /root/pb_done2 && test -f /root/vc/data/dl_done && break; sleep 10; done
cp /root/*_pb.h5ad /root/vc/data/
cd /root/vc
export PYTHONIOENCODING=utf-8
python src/offset_pairs.py --seeds 20 \
  --pairs hepg2:jurkat,jurkat:hepg2,hepg2e:hepg2o,hepg2o:hepg2e,ess:hepg2,ess:jurkat,rpe1:hepg2 \
  --out nadig_pairs_metrics.json > results/nadig_pairs.log 2>&1
echo FIN > /root/analysis_done
