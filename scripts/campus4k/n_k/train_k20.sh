#！/bin/bash

python tools/train.py --config configs/campus4k/n_k/n20_k20.yml -g 3
sleep 5

python tools/train.py --config configs/campus4k/n_k/n50_k20.yml -g 3
sleep 5

python tools/train.py --config configs/campus4k/n_k/n100_k20.yml -g 3
sleep 5

python tools/train.py --config configs/campus4k/n_k/n200_k20.yml -g 3
sleep 5