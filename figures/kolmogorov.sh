#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

echo "Kolomogorov Flow: linear vs. nonlinear"
cd ../example/KolmogorovFlow

filter_types=("linear" "nonlinear")
noise_level=1
sensor_every=1

cp data/test.npz ../../figures/data/kolmogorov.npz
for i in {0..1}
do
    filter_type=${filter_types[$i]}
    python main.py --filter_type=$filter_type --sensor_every=$sensor_every \
    --noise_level=$noise_level --include_training=True

    cp data/KF_${filter_type}_Noise${noise_level}Obs${sensor_every}Rank20_test.npz ../../figures/data/kolmogorov_${filter_type}.npz
done

cd ../../figures
julia kolmogorov.jl