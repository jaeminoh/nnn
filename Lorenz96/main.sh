#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export JAX_ENABLE_X64=True

for u in 60
do
    for lr0 in 1e-3 1e-4
    do
        for epoch in 10000
        do
            for noise in 0 1 5
            do
                python lorenz96_acceleration.py --unroll_length=$u --noise_level=$noise --lr0=$lr0 --epoch=$epoch > results/unroll"$u"_noise"$noise"_lr0"$lr0"_epoch"$epoch".txt
            done
        done
    done
done