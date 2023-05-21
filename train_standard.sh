#!/bin/bash
mkdir -p checkpoints
python -u train.py --name CEDlow --stage chairs --validation chairs --gpus 0 --num_steps 350000 --batch_size 4 --lr 0.00025 --image_size 368 496 --wdecay 0.0001
