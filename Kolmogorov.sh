#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5

cd example/KolmogorovFlow

for noise in 0.5 0.75 1.0 
do
    for sensor_every in 1 2 4
    do
        for filter_type in "nonlinear" "linear"
        do
        python main.py --filter_type=$filter_type --sensor_every=$sensor_every \
        --noise_level=$noise --include_training=True 
        done
    done
done

