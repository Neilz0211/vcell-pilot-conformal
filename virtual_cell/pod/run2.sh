#!/usr/bin/env bash
# Pipeline-equivalence check + 4-line matrix: reprocess K562-essential raw single cell with pod/pseudobulk.py.
cd /root/d && wget -q https://ndownloader.figshare.com/files/35773219 -O k562ess_raw.h5ad && echo dl > /root/dl2.log
cd /root && python pseudobulk.py d/k562ess_raw.h5ad /root/vc/data/k562mine_pb.h5ad > pb_k562.log 2>&1
cd /root/vc && export PYTHONIOENCODING=utf-8
python src/offset_pairs.py --seeds 20 \
  --pairs ess:k562m,k562m:hepg2,hepg2:k562m,k562m:jurkat,jurkat:k562m,rpe1:jurkat,jurkat:rpe1,hepg2:rpe1 \
  --out matrix_pairs_metrics.json > results/matrix_pairs.log 2>&1
echo FIN > /root/analysis2_done
