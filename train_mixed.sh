#!/bin/bash
mkdir -p checkpoints
python -u train.py --name CEDFlow --stage chairs --validation chairs --gpus 0 --num_steps 400000 --batch_size 4 --lr 0.0003 --image_size 368 496 --wdecay 0.0001 --mixed_precision
