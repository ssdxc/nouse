#！/bin/bash

python tools/train.py --config configs/campus4k/n_k/n20_k5.yml -g 1
sleep 5

python tools/train.py --config configs/campus4k/n_k/n50_k5.yml -g 1
sleep 5

python tools/train.py --config configs/campus4k/n_k/n100_k5.yml -g 1
sleep 5

python tools/train.py --config configs/campus4k/n_k/n200_k5.yml -g 1
sleep 5