#！/bin/bash

python tools/train.py --config configs/campus4k/n_k/n20_k10.yml -g 2
sleep 5

python tools/train.py --config configs/campus4k/n_k/n50_k10.yml -g 2
sleep 5

python tools/train.py --config configs/campus4k/n_k/n100_k10.yml -g 2
sleep 5

python tools/train.py --config configs/campus4k/n_k/n200_k10.yml -g 2
sleep 5