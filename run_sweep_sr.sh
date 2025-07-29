#!/bin/bash

SR_LIST=(0.6 0.8 1.0 1.2 1.4)
ALPHA_LEN=30
OUTDIR="reservoir_states_data2"

for SR in "${SR_LIST[@]}"
do
    echo "==== Running spectrum_radius=$SR ===="
    python reservoir_states_simulate.py \
        --output_dir "$OUTDIR" \
        --spectrum_radius "$SR" \
        --alpha_len "$ALPHA_LEN"
    python compute_custom_esn_capacity.py \
        --data_dir "$OUTDIR" \
        --spectrum_radius "$SR" \
        --alpha_len "$ALPHA_LEN"
done 

SR=0.8
ALPHA_LEN=30
OUTDIR="reservoir_states_data2"

nohup python compute_custom_esn_capacity.py \
    --data_dir "$OUTDIR" \
    --spectrum_radius "$SR" \
    --alpha_len "$ALPHA_LEN" \
    > compute_capacity_${SR}.log 2>&1 &