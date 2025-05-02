#!/bin/bash

XLA_PYTHON_CLIENT_MEM_FRACTION=.20
CUDA_VISIBLE_DEVICES=7

cd example/Lorenz96
#python L96.py --epoch=50 --noise_level=5 --sensor_every=1
python main.py --epoch=100 --noise_level=5 --sensor_every=1