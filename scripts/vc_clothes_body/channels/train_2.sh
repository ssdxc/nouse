#！/bin/bash

python tools/train.py --config configs/vc_clothes_body/channels/cos_32_32_none_2_1_-3_0_02.yml -g 1
sleep 5

python tools/train.py --config configs/vc_clothes_body/channels/cos_32_16_none_2_1_-3_0_02.yml -g 1
sleep 5

python tools/train.py --config configs/vc_clothes_body/channels/cos_32_8_none_2_1_-3_0_02.yml -g 1
sleep 5

python tools/train.py --config configs/vc_clothes_body/channels/cos_16_32_none_2_1_-3_0_02.yml -g 1
sleep 5

python tools/train.py --config configs/vc_clothes_body/channels/cos_16_16_none_2_1_-3_0_02.yml -g 1
sleep 5

python tools/train.py --config configs/vc_clothes_body/channels/cos_16_8_none_2_1_-3_0_02.yml -g 1
sleep 5

python tools/train.py --config configs/vc_clothes_body/channels/cos_8_32_none_2_1_-3_0_02.yml -g 1
sleep 5

python tools/train.py --config configs/vc_clothes_body/channels/cos_8_16_none_2_1_-3_0_02.yml -g 1
sleep 5

python tools/train.py --config configs/vc_clothes_body/channels/cos_8_8_none_2_1_-3_0_02.yml -g 1
sleep 5