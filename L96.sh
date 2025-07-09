#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5
#export JAX_ENABLE_X64=True

rank=20
epoch=300
unroll_length=5 # 3 5 15
filter_type="linear"

Nxs=(40 128)
forcings=(4 8 16)
noise_levels=(0.1854 0.364 0.6298)

cd example/Lorenz96
for Nx in 40 128
do
    for i in {0..2}
    do
    forcing=${forcings[$i]}
    noise_level=${noise_levels[$i]}
    python make_data.py --Nx=$Nx --forcing=$forcing
        for sensor_every in 1 2 4
        do
        python main.py --filter_type=$filter_type \
        --forcing=$forcing --noise_level=$noise_level \
        --sensor_every=$sensor_every --Nx=$Nx --rank=$rank \
        --include_training=True --epoch=$epoch --unroll_length=$unroll_length
        done
    done
done