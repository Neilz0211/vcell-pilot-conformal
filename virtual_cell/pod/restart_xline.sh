#!/usr/bin/env bash
pkill -f tahoe_xline.py; sleep 2
cd /root/code && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python tahoe_xline.py /root/tahoe --seeds 5 --workers 6 --out /root/tahoe_xline.json > /root/xline.log 2>&1 && echo XLINE_OK >> /root/xline.log
