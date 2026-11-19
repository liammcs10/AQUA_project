#!/bin/bash
# N250 sims (random init + 500 ms warm-up at 250 pA^2), then PSTH edivisive / esst. Run from changepoint_analysis/.
set -e
PY=/home/liam/anaconda3/envs/aqua/bin/python
for C in mean variance; do
  echo "=== simulate $C $(date)"
  for r in "0 63" "63 126" "126 189" "189 250"; do
    CHANGE=$C $PY CPA_with_AQUA.py $r > sim_N250_${C}_r${r// /-}.log 2>&1 &
  done
  for j in $(jobs -p); do wait $j; done   # fail the script if any part failed
  CHANGE=$C $PY CPA_with_AQUA.py merge
done
for C in mean variance; do
  for m in edivisive esst; do
    echo "=== psth $C $m $(date)"
    $PY -c "from CPA_with_AQUA import psth_cusum_analysis as p; p('CPA_cusum_${C}_N250_spikes.pickle', 'CPA_psth_${C}_N250_${m}_bin25.pickle', method='$m', bin_ms=25, t_discard=500)" > psth_N250_${C}_${m}.log 2>&1
  done
done
echo "=== done $(date)"
