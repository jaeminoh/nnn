#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
#export JAX_ENABLE_X64=True



for noise in 0 1 5 10
do
    python base.py --noise_level=$noise
done