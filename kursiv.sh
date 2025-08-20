export CUDA_VISIBLE_DEVICES=6
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Kursiv: linear vs. nonlinear"

filter_types=("linear" "nonlinear")

noise=0.5
sensor_every=4
method="etdrk4"

python example/kursiv/make_data.py --draw_plot=True
for i in {0..1}
do
    filter_type=${filter_types[$i]}
    #method=${methods[$i]}
    #inner_step=${inner_steps[$i]}
    python example/kursiv/main.py --filter_type=$filter_type --method=$method --sensor_every=$sensor_every \
    --noise_level=$noise --include_training=True
    #cp data/Kursiv_${filter_type}_Noise${noise}Obs${sensor_every}Rank20_test2.npz ../../figures/data/kursiv_${filter_type}.npz
done