#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5
#export JAX_ENABLE_X64=True

echo "Kursiv: linear vs. nonlinear"

filter_types=("linear" "nonlinear")

cd ../example/Kursiv
noise=0.5
sensor_every=1
method="etdrk4"
inner_step=10

python make_data.py
for i in {0..1}
do
    filter_type=${filter_types[$i]}
    python main.py --filter_type=$filter_type --method=$method --sensor_every=$sensor_every \
    --inner_steps=$inner_step --noise_level=$noise --include_training=True

    cp data/Kursiv_${filter_type}_Noise${noise}Obs${sensor_every}Rank20_test2.npz ../../figures/data/kursiv_${filter_type}.npz
done

