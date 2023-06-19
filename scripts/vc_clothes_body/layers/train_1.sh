#！/bin/bash

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_1_2_-3_0_02.yml -g 0
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_1_1_-3_0_02.yml -g 0
sleep 5
