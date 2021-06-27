python main.py --dataset cifar100 --num-labeled $1 --out $2 --num-super $3 --arch wideresnet --lambda_oem 0.1 --lambda_socr 1.0 \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --opt_level O2 --amp --mu 2







