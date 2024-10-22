#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
#export JAX_ENABLE_X64=True



for lr0 in 1e-3 1e-4
    do
    for epoch in 5000
    do
        for noise in 0
        do
            python baseline.py --noise_level=$noise --lr0=$lr0 --epoch=$epoch > results/noise"$noise"_lr"$lr0"_epoch"$epoch".txt
        done
    done
done