#!/bin/sh
for entry in RUNS/*
do
        python segment.py --input_folder /home/stefan/Desktop/ruemonge2014/dataset --output_folder "$entry"/segmentation/ --logdir "$entry"
done

