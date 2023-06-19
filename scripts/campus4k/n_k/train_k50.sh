#！/bin/bash

python tools/train.py --config configs/campus4k/n_k/n50_k50.yml -g 4
sleep 5

python tools/train.py --config configs/campus4k/n_k/n100_k50.yml -g 4
sleep 5

python tools/train.py --config configs/campus4k/n_k/n200_k50.yml -g 4
sleep 5