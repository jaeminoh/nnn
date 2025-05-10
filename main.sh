#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5
#export JAX_ENABLE_X64=True

export Nx=128

cd example/Lorenz96
python make_data.py --Nx=$Nx --forcing=16
python main.py --Nx=$Nx --epoch=300 --noise_level=63 --sensor_every=1 --rank=20 --include_training=True