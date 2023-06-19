#！/bin/bash

python tools/train.py --config configs/campus4k/n_k/n100_k100.yml -g 5
sleep 5

python tools/train.py --config configs/campus4k/n_k/n200_k100.yml -g 5
sleep 5