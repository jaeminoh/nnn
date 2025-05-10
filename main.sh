#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5
#export JAX_ENABLE_X64=True

Nx=128
forcings=(4 8 16)
noise_levels=(0.1854 0.364 0.6298)

cd example/Lorenz96

for i in {0..2}
do
    forcing=${forcings[$i]}
    noise_level=${noise_levels[$i]}
    for sensor_every in 1 2 4
    do
    echo "Nx: $Nx"
    echo "forcing: $forcing"
    echo "noise_level: $noise_level"
    echo sensor_every=$sensor_every
    python make_data.py --Nx=$Nx --forcing=$forcing
    python main.py --Nx=$Nx --epoch=300 --noise_level=$noise_level --sensor_every=$sensor_every --rank=20 --include_training=True      
    done
done

