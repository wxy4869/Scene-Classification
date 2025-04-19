#!/bin/bash

python scene_recog_cnn.py  --phase train --train_data_dir ./data/train --model_dir ./output/trained_cnn.pth
for i in {1..10}
do
    python util.py
    python scene_recog_cnn.py --phase test --test_data_dir ./data/tmp --model_dir ./output/trained_cnn.pth >> ./log/test.log 2>&1
done