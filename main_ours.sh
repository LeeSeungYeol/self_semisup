#!/bin/bash
#data="voc_1over32"
data=$1
## Train a supervised model with labelled images
#python3 main.py --config configs/${data}_baseline.json
## Train a semi-supervised model with both labelled and unlabelled images
python3 main.py --config configs/${data}_ours.json
