#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export JAX_ENABLE_X64=True

for length in 60
do
    for noise in 0 1 5
    do
        python ncde.py --noise_level=$noise --length=$length
    done
done