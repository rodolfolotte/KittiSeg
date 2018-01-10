"""
Evaluate CNN models for facade features detection 

Utilizes: Trained FacadeSeg weights. If no logdir is given, fails.

Usage:  
 python evaluate_facade.py
    --logdir /home/rodolfo/Dropbox/phd/results/facades-benchmark/ruemonge2014/facadeSeg/FacadeSeg_VGG_2017_10_31_13.00/

 tv-analyze --logdir /home/rodolfo/Dropbox/phd/results/facades-benchmark/ruemonge2014/facadeSeg/FacadeSeg_VGG_2017_10_31_13.00/

--------------------------------------------------------------------------------
This is an extension code from KittSeg

Author: MarvinTeichmann

Link: https://github.com/MarvinTeichmann/KittiSeg
--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Rodolfo Lotte

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'submodules')

import tensorvision.train as train
import tensorvision.analyze as ana
import tensorvision.utils as utils

#from evaluation import facade_test

flags.DEFINE_string('hypes', 'hypes/FacadeSeg_VGG.json', 'File storing model parameters.')

if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))

def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
        import tensorflow_fcn.utils
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    hypes_path = FLAGS.logdir
    hypes_path = os.path.join(hypes_path, "model_files/hypes.json")

    with open(hypes_path, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        runs_dir = os.path.join(os.environ['TV_DIR_RUNS'], 'FacadeSeg')
    else:
        runs_dir = 'RUNS'

    utils.set_dirs(hypes, FLAGS.hypes)
    utils._add_paths_to_sys(hypes)

    logging.info("Evaluating on Validation data.")    
    ana.do_analyze(FLAGS.logdir)

    logging.info("Segmenting and test data. Creating output.")
    ana.do_inference(FLAGS.logdir)

    logging.info("Analysis for pretrained model complete.")
    

if __name__ == '__main__':
    tf.app.run()
