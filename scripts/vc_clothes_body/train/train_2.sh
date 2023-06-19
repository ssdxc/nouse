#！/bin/bash

python tools/train.py --config configs/vc_clothes_body/train/edge_32_32_relu_2_1_-3_-4_02.yml -g 4
sleep 5

python tools/train.py --config configs/vc_clothes_body/train/edge_32_32_relu_2_1_-3_-4_05.yml -g 4
sleep 5

python tools/train.py --config configs/vc_clothes_body/train/edge_32_32_relu_2_1_-3_0_02.yml -g 4
sleep 5

python tools/train.py --config configs/vc_clothes_body/train/edge_32_32_relu_2_1_-3_0_05.yml -g 4
sleep 5

python tools/train.py --config configs/vc_clothes_body/train/edge_32_32_relu_2_1_-4_0_02.yml -g 4
sleep 5

python tools/train.py --config configs/vc_clothes_body/train/edge_32_32_relu_2_1_-4_0_05.yml -g 4
sleep 5

python tools/train.py --config configs/vc_clothes_body/train/edge_32_32_relu_2_1_-4_-4_02.yml -g 4
sleep 5

python tools/train.py --config configs/vc_clothes_body/train/edge_32_32_relu_2_1_-4_-4_05.yml -g 4
sleep 5