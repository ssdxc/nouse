#!/bin/bash

python tools/train_v3.py --config configs/main/vc_clothes/ablation_bot_sef_indep.yml -g 3

python tools/train_v3.py --config configs/main/vc_clothes/ablation_bot_sef.yml -g 3

python tools/train_v3.py --config configs/main/vc_clothes/ablation_bot_sef_indep.yml -g 3

##

python tools/train_v3.py --config configs/main/vc_clothes/ablation_bot_sef_indep.yml -g 3

python tools/train_v3.py --config configs/main/vc_clothes/ablation_bot_sef.yml -g 3

python tools/train_v3.py --config configs/main/vc_clothes/ablation_bot_sef_indep.yml -g 3

##

python tools/train_v3.py --config configs/main/campus4k/ablation_bot_sef_indep.yml -g 3

python tools/train_v3.py --config configs/main/campus4k/ablation_bot_sef.yml -g 3

python tools/train_v3.py --config configs/main/campus4k/ablation_bot_sef_indep.yml -g 3