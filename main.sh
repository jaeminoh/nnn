#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export CUDA_VISIBLE_DEVICES=5
#export JAX_ENABLE_X64=True

cd example/Lorenz96
#python L96.py --epoch=50 --noise_level=5 --sensor_every=1
python main.py --epoch=300 --noise_level=100 --sensor_every=2 --rank=20 --include_training=False