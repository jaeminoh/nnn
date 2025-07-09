#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5

echo "Lorenz96: linear vs. nonlinear"
cd ../example/Lorenz96

Nx=40
forcing=16
noise_level=0.6298
sensor_every=4

python make_data.py --Nx=$Nx --forcing=$forcing
cp data/test.npz ../../figures/data/L96.npz

for filter_type in "linear" "nonlinear"
do
    python main.py --filter_type=$filter_type --forcing=$forcing --noise_level=$noise_level \
    --sensor_every=$sensor_every --Nx=$Nx --include_training=True --epoch=100

    cp data/L96_${filter_type}_Forcing${forcing}Noise${noise_level}Obs${sensor_every}Nx${Nx}_test.npz ../../figures/data/L96_${filter_type}.npz
done

cd ../../figures
julia L96.jl