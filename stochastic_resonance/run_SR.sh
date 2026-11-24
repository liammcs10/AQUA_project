#!/bin/bash
# Run white-noise then OU stochastic resonance sims. Run from stochastic_resonance/:
#   nohup ./run_SR.sh > run_SR.log 2>&1 &
set -e
cd "$(dirname "$0")"
PY=/home/liam/anaconda3/envs/aqua/bin/python
for N in white OU; do
  echo "=== $N $(date)"
  NOISE_TYPE=$N $PY stochastic_resonance.py > SR_${N}.log 2>&1
done
echo "=== done $(date)"
