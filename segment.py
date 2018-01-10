"""
Segment Facade Features in an image using FacadeSeg.

Utilizes: Trained FacadeSeg weights. If no logdir is given, fails.

Usage:
   python segment.py 
      --input_folder PATH/dataset/ 
      --output_folder PATH/segmentation/
      --logdir PATH/FacadeSeg_RUN/
      --gpus 4

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'submodules')

from evaluation import seg_utils as seg

try:
    # Check whether setup was done correctly
    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_folder', None,
                    'Folder to apply FacadeSeg.')
flags.DEFINE_string('output_folder', None,
                    'Folder to save FacadeSeg results.')

def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input_folder is None:
        logging.error("No input_image was given.")
        logging.info(
            "Usage: python segment.py [--input_folder /path/to/data/] "
            "[--output_folder /path/to/result/] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")

        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from FacadeSeg
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'], 'FacadeSeg')
        else:
            runs_dir = 'RUNS'
        
        logdir = os.path.join(runs_dir, "FacadeSegPreTrained")
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')
    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules, image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    # classes
    classes_colors =  [[0, 0, 255], 
                       [0, 255, 255], 
                       [255, 255, 0], 
		       [255, 0, 255],
                       [255, 0, 0],                         
		       [0, 0, 0],
		       [255, 128, 0],
                       [0, 255, 0]]

    input_folder = FLAGS.input_folder
    output_folder = FLAGS.output_folder

    # If the folder is not empty
    if(not os.listdir(input_folder)==""):
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                input_image = os.path.join(input_folder, file)

                logging.info("Starting inference using {} as input".format(input_image))

                # Load and resize input image
                image = scp.misc.imread(input_image)

                if hypes['jitter']['reseize_image']:
                    # Resize input only, if specified in hypes
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    image = scp.misc.imresize(image, size=(image_height, image_width), interp='cubic')
            

                # Run FacadeSeg model on image
                feed = {image_pl: image}
                softmax = prediction['softmax']
                logits = prediction['logits']
                output, lll = sess.run([softmax, logits], feed_dict=feed)

                # Reshape output from flat vector to 2D Image
                shape = image.shape
                output_image = output.reshape(shape[0], shape[1], -1)

                x = np.argmax(output_image, axis=2)
                im = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
                for i,_ in enumerate(x):
                    for j,_ in enumerate(x[i]):
                        value = x[i][j]
                        color_code = classes_colors[value]
                        im[i][j] = color_code


                # Save output images to disk.
                if(not os.path.isdir(output_folder)):
                    logging.info("Output directory does not exist. Creating in: " + output_folder)
                    os.makedirs(output_folder)
                
                raw_image_name = file.split('.')[0] + '.png'
                full_raw_image_name = os.path.join(output_folder, raw_image_name)

                scp.misc.imsave(full_raw_image_name, im)
                logging.info("Labelled image saved in " + full_raw_image_name + "\n")   


    else:
        logging.info("Input folder is empty. Check it again")


if __name__ == '__main__':
    tf.app.run()
