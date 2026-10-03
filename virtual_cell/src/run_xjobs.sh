#!/usr/bin/env bash
# Run the D1/D2 exploratory jobs pair by pair in parallel with limited BLAS threads (shared machine).
# usage: bash src/run_xjobs.sh [P=6]
P=${1:-6}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
L="ess rpe1 hepg2 jurkat"
{
  for B in $L; do echo "xmulti.py --seeds 20 --ms 25,50,100,200 --only $B|xmulti_$B"; done
  for A in $L; do for B in $L; do [ $A != $B ] && echo "xmap.py --seeds 20 --ms 50,100,200 --only $A,$B|xmap_$A-$B"; done; done
  for A in $L; do for B in $L; do [ $A != $B ] && echo "xdesign.py --seeds 20 --mfit 25,50,100 --only $A,$B|xdesign_$A-$B"; done; done
} | xargs -P "$P" -I{} bash -c 'j="{}"; cmd="${j%%|*}"; log="${j##*|}"; uv run python src/$cmd > results/xlogs/$log.log 2>&1; echo "done $log"'
echo ALL_XJOBS_DONE
