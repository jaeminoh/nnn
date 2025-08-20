#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5
#export JAX_ENABLE_X64=True

Nx=40
forcing=8
noise_level=0.364

cd example/Lorenz96
python make_data.py --Nx=$Nx --forcing=$forcing
for unroll_length in 1 3 5 15
do
    for sensor_every in 1 2 4
    do
    echo sensor_every=$sensor_every
    echo unroll_length=$unroll_length
    python main.py --Nx=$Nx --epoch=300 --noise_level=$noise_level --sensor_every=$sensor_every --rank=20 --include_training=True --unroll_length=$unroll_length --save_dir=./lorenz96_unroll_length_$unroll_length\_sensor_every_$sensor_every    
    done
done

