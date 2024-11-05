#!/bin/bash


export CUDA_VISIBLE_DEVICES=6
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
#export JAX_DEFAULT_MATMUL_PRECISION=float32


for epoch in 10000
do
    for noise in 250
    do
        python base_ensembles.py --rank=256 --epoch=$epoch --noise_level=$noise --include_training=True
    done
done
