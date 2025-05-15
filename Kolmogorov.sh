#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=7
#export JAX_ENABLE_X64=True

rank=20
epoch=300
unroll_length=10

cd example/KolmogorovFlow
noises=(0.5 0.75 1.0)

for noise in "${noises[@]}"
do
    for sensor_every in 1 2 4
    do
    python main.py \
    --sensor_every=$sensor_every --rank=$rank --noise_level=$noise \
    --include_training=True --epoch=$epoch --unroll_length=$unroll_length
    done
done

