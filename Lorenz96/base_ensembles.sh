#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
#export JAX_DEFAULT_MATMUL_PRECISION=float32


for epoch in 1000 5000 10000 50000
do
    python base_ensembles.py --rank=256 --epoch=$epoch --noise_level=5
done
