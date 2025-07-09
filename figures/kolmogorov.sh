#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

echo "Kolomogorov Flow: linear vs. nonlinear"
cd ../example/KolmogorovFlow

noise_level=1.0
sensor_every=4

for filter_type in "nonlinear" "linear"
do
    python main.py --filter_type=$filter_type --sensor_every=$sensor_every \
    --noise_level=$noise_level --include_training=True

    cp data/KF_${filter_type}_Noise${noise_level}Obs${sensor_every}Rank20_test.npz ../../figures/data/kolmogorov_${filter_type}.npz
done

cd ../../figures
julia kolmogorov.jl