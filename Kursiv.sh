#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5


filter_type="linear"
epoch=300

cd example/Kursiv
noises=(0.25 0.375 0.5)
method="forward_euler" #"etdrk4"
inner_step=25 #10


python make_data.py
for noise in "${noises[@]}"
do
    for sensor_every in 1 2 4
    do
    python main.py --filter_type=$filter_type --method=$method --inner_steps=$inner_step \
    --sensor_every=$sensor_every  --noise_level=$noise --include_training=True --epoch=$epoch
    done
done
