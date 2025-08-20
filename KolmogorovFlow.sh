export CUDA_VISIBLE_DEVICES=6
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Kolomogorov Flow: linear vs. nonlinear"

noise_level=1.0
sensor_every=4

JAX_ENABLE_X64=True python example/KolmogorovFlow/make_data.py
for filter_type in "nonlinear" "linear"
do
    python example/KolmogorovFlow/main.py --filter_type=$filter_type --sensor_every=$sensor_every \
    --noise_level=$noise_level --include_training=True
done