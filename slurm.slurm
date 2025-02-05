#!/bin/bash
# NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=oda
#SBATCH --time=05:30:00
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --output=slurm.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1

ml purge
cd $SCRATCH/ODA
source .venv/bin/activate
cd example/Lorenz96
python main.py --epoch=50 --noise_level=50 --sensor_every=2