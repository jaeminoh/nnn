#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

for u in 20 40 60 80 100
do
    for lr0 in 1e-2 1e-3 1e-4 
    do
        python lorenz96_acceleration.py --unroll_length=$u --lr0=$lr0 > outputs/unroll"$u"_lr0"$lr0".txt
    done
done