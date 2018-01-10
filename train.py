"""
Trains, evaluates and saves the KittiSeg model.

-------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

More details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import logging
import os
import sys
import collections
import commentjson

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'submodules')

import tensorvision.train as train
import tensorvision.utils as utils

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', None,
                    'File storing model parameters.')

flags.DEFINE_string('mod', None,
                    'Modifier for model parameters.')

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


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def main(_):  
    logging.info("Initializing GPUs, plugins and creating the essential folders")
    utils.set_gpus_to_use()

    if FLAGS.hypes is None:
        logging.error("No hypes are given.")
        logging.error("Usage: python train.py --hypes hypes.json")
        logging.error("   tf: tv-train --hypes hypes.json")
        exit(1)

    with open(FLAGS.hypes) as f:
        logging.info("f: %s", f)                
        hypes = commentjson.load(f)
            
    if FLAGS.mod is not None:
        import ast
        mod_dict = ast.literal_eval(FLAGS.mod)
        dict_merge(hypes, mod_dict)
    
    logging.info("Loading plugins")
    utils.load_plugins()

    logging.info("Set dirs")
    utils.set_dirs(hypes, FLAGS.hypes)

    logging.info("Add paths to sys")
    utils._add_paths_to_sys(hypes)
    
    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)

    tf.reset_default_graph()
    
    logging.info("Start training")    
    train.do_training(hypes)


if __name__ == '__main__':    
    tf.app.run()
