#！/bin/bash

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_1_1_-3_0_02.yml -g 2
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_1_2_-3_0_02.yml -g 2
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_1_3_-3_0_02.yml -g 2
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_2_1_-3_0_02.yml -g 2
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_2_2_-3_0_02.yml -g 2
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_2_3_-3_0_02.yml -g 2
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_3_1_-3_0_02.yml -g 2
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_3_2_-3_0_02.yml -g 2
sleep 5

python tools/train.py --config configs/vc_clothes_body/layers/edge_32_32_none_3_3_-3_0_02.yml -g 2
sleep 5