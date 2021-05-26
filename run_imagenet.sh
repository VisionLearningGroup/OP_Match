#!/bin/bash
#SBATCH --partition=gpu-L --gres=gpu:1 --constraint="gpu4" --cpus-per-gpu=10

d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/semi_open
source activate pytorch
python main.py --dataset cifar10 --num-labeled $1 --out $2 --arch resnet_imagenet --lambda_oem 0.1 --lambda_socr 1.0 \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --opt_level O2 --amp --mu 2







