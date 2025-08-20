#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5


filter_type="linear"
epoch=300

cd example/Kursiv
noises=(0.25 0.375 0.5)
method="etdrk4"
inner_step=10


python make_data.py
for noise in 0.25 0.375 0.5
do
    for sensor_every in 1 2 4
    do
        for filter_type in "linear" "nonlinear"
        do
            python main.py --filter_type=$filter_type --method=$method \
            --sensor_every=$sensor_every  --noise_level=$noise --include_training=True
        done
    done
done
