export CUDA_VISIBLE_DEVICES=6
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Lorenz96: linear vs. nonlinear"

Nx=40
forcing=16
noise_level=0.6298
sensor_every=4

python example/Lorenz96/make_data.py --Nx=$Nx --forcing=$forcing --draw_plot=True
for filter_type in "linear" "nonlinear"
do
    python example/Lorenz96/main.py --filter_type=$filter_type --forcing=$forcing --noise_level=$noise_level \
    --sensor_every=$sensor_every --Nx=$Nx --include_training=True --epoch=100    
done
