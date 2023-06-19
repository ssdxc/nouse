#！/bin/bash

python tools/train.py --config configs/campus4k/n_k/n20_k2.yml -g 0
sleep 5

python tools/train.py --config configs/campus4k/n_k/n50_k2.yml -g 0
sleep 5

python tools/train.py --config configs/campus4k/n_k/n100_k2.yml -g 0
sleep 5

python tools/train.py --config configs/campus4k/n_k/n200_k2.yml -g 0
sleep 5