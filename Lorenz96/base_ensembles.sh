#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
#export JAX_DEFAULT_MATMUL_PRECISION=float32


for epoch in 5000 10000
do
    python base_ensembles.py --rank=256 --epoch=$epoch --noise_level=35 --include_training=True
done
