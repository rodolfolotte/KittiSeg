"""
An implementation of FCN in tensorflow.
------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import scipy as scp
import random
import logging

import tensorflow as tf

from evaluation import seg_utils as seg
from tensorflow.python.framework import dtypes
from math import ceil

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)


def _add_softmax(hypes, logits):
    num_classes = hypes['arch']['num_classes']
    with tf.name_scope('decoder'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=hypes['solver']['epsilon'])
        # logits = logits + epsilon

        softmax = tf.nn.softmax(logits)

    return softmax


def decoder(hypes, logits, train=True, skip=True, debug=False):
    fcn_in = logits['fcn_in']
    num_classes = hypes['arch']['num_classes']

    fcn_in = tf.Print(fcn_in, [tf.shape(fcn_in)],
                      message='Shape of %s' % fcn_in.name,
                      summarize=4, first_n=1)

    if 'scale_down' in hypes:
        sd = hypes['scale_down']
    else:
        sd = 1

    he_init = tf.contrib.layers.variance_scaling_initializer()
    l2_regularizer = tf.contrib.layers.l2_regularizer(hypes['wd'])

    # Build score_fr layer
    score_fr = tf.layers.conv2d(
        fcn_in, kernel_size=[1, 1], filters=num_classes, padding='SAME',
        name='score_fr', kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer)

    _activation_summary(score_fr)

    # Do first upsampling
    upscore2 = _upscore_layer(
        score_fr, upshape=tf.shape(logits['feed2']),
        num_classes=num_classes, name='upscore2', ksize=4, stride=2)

    he_init2 = tf.contrib.layers.variance_scaling_initializer(factor=2.0*sd)

    # Score feed2
    score_feed2 = tf.layers.conv2d(
        logits['feed2'], kernel_size=[1, 1], filters=num_classes,
        padding='SAME', name='score_feed2', kernel_initializer=he_init2,
        kernel_regularizer=l2_regularizer)

    _activation_summary(score_feed2)

    if skip:
        # Create skip connection
        fuse_feed2 = tf.add(upscore2, score_feed2)
    else:
        fuse_feed2 = upscore2
        fuse_feed2.set_shape(score_feed2.shape)

    # Do second upsampling
    upscore4 = _upscore_layer(
        fuse_feed2, upshape=tf.shape(logits['feed4']),
        num_classes=num_classes, name='upscore4', ksize=4, stride=2)

    he_init4 = tf.contrib.layers.variance_scaling_initializer(factor=2.0*sd*sd)
    # Score feed4
    score_feed4 = tf.layers.conv2d(
        logits['feed4'], kernel_size=[1, 1], filters=num_classes,
        padding='SAME', name='score_feed4', kernel_initializer=he_init4,
        kernel_regularizer=l2_regularizer)

    _activation_summary(score_feed4)

    if skip:
        # Create second skip connection
        fuse_pool3 = tf.add(upscore4, score_feed4)
    else:
        fuse_pool3 = upscore4
        fuse_pool3.set_shape(score_feed4.shape)

    # Do final upsampling
    upscore32 = _upscore_layer(
        fuse_pool3, upshape=tf.shape(logits['images']),
        num_classes=num_classes, name='upscore32', ksize=16, stride=8)

    decoded_logits = {}
    decoded_logits['logits'] = upscore32
    decoded_logits['softmax'] = _add_softmax(hypes, upscore32)

    return decoded_logits


def _upscore_layer(bottom, upshape, num_classes, name, ksize=4, stride=2):
    strides = [1, stride, stride, 1]

    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        new_shape = [upshape[0], upshape[1], upshape[2], num_classes]
        output_shape = tf.stack(new_shape)

        f_shape = [ksize, ksize, num_classes, in_features]

        up_init = upsample_initilizer()

        weights = tf.get_variable(name="weights", initializer=up_init, shape=f_shape)

        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)

        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        deconv = tf.Print(deconv, [tf.shape(deconv)],
                          message='Shape of %s' % name,
                          summarize=4, first_n=1)

        _activation_summary(deconv)

    return deconv


def upsample_initilizer(dtype=dtypes.float32):
    """Returns an initializer that creates filter for bilinear upsampling.

    Use a transposed convolution layer with ksize = 2n and stride = n to
    perform upsampling by a factor of n.
    """
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-float point type.')

    def _initializer(shape, dtype=dtype, partition_info=None):
        """Initializer function."""
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating type.')

        width = shape[0]
        heigh = shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([shape[0], shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(shape)
        
        for i in range(shape[2]):
            '''
            the next line of code is correct as given
            [several issues were opened ...]
            we only want to scale each feature,
            so there is no interaction between channels,
            that is why only the diagonal i, i is initialized
            '''
            weights[:, :, i, i] = bilinear

        return weights

    return _initializer


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def loss(hypes, decoded_logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    logits = decoded_logits['logits']
    num_classes = hypes['arch']['num_classes']
    with tf.name_scope('loss'):

        logits = tf.reshape(logits, (-1, num_classes))
        shape = [logits.get_shape()[0], num_classes]
        epsilon = tf.constant(value=hypes['solver']['epsilon'])
        # logits = logits + epsilon
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if hypes['loss'] == 'xentropy':
            cross_entropy_mean = _compute_cross_entropy_mean(hypes, labels,
                                                             softmax)
        elif hypes['loss'] == 'softF1':
            cross_entropy_mean = _compute_f1(hypes, labels, softmax, epsilon)

        elif hypes['loss'] == 'softIU':
            cross_entropy_mean = _compute_soft_ui(hypes, labels, softmax,
                                                  epsilon)

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

        weight_loss = tf.add_n(tf.get_collection(reg_loss_col),
                               name='reg_loss')

        total_loss = cross_entropy_mean + weight_loss

        losses = {}
        losses['total_loss'] = total_loss
        losses['xentropy'] = cross_entropy_mean
        losses['weight_loss'] = weight_loss

    return losses


def _compute_cross_entropy_mean(hypes, labels, softmax):
    head = hypes['arch']['weight']
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), head),
                                   reduction_indices=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='xentropy_mean')
    return cross_entropy_mean


def _compute_f1(hypes, labels, softmax, epsilon):
    num_classes = hypes['arch']['num_classes']
    labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))[:, 1]    
    logits = softmax[:, 1]
    true_positive = tf.reduce_sum(labels*logits)
    false_positive = tf.reduce_sum((1-labels)*logits)

    recall = true_positive / tf.reduce_sum(labels)
    precision = true_positive / (true_positive + false_positive + epsilon)

    score = 2*recall * precision / (precision + recall)
    f1_score = 1 - 2*recall * precision / (precision + recall)

    return f1_score


def _compute_soft_ui(hypes, labels, softmax, epsilon):
    intersection = tf.reduce_sum(labels*softmax, reduction_indices=0)
    union = tf.reduce_sum(labels+softmax, reduction_indices=0) \
        - intersection+epsilon

    mean_iou = 1-tf.reduce_mean(intersection/union, name='mean_iou')

    return mean_iou


# Bendidi> https://github.com/MarvinTeichmann/KittiSeg/issues/15
def evaluation(hypes, images, labels, decoded_logits, losses, global_step):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    eval_list = []
    num_classes = hypes['arch']['num_classes']
    logits = tf.reshape(decoded_logits['logits'], (-1, num_classes))
    labels = tf.reshape(labels, (-1, num_classes))
    pred = tf.argmax(logits, dimension=1)
    y = tf.argmax(labels, 1)
    Prec = []
    Rec = []
    f1 = []
    for i in range(num_classes):
        tp = tf.count_nonzero(tf.cast(tf.equal(pred,i),tf.int32) * tf.cast(tf.equal(y,i),tf.int32))
        tn = tf.count_nonzero(tf.cast(tf.not_equal(pred,i),tf.int32) * tf.cast(tf.not_equal(y,i),tf.int32))
        fp = tf.count_nonzero(tf.cast(tf.equal(pred,i),tf.int32) * tf.cast(tf.not_equal(y,i),tf.int32))
        fn = tf.count_nonzero(tf.cast(tf.not_equal(pred,i),tf.int32) * tf.cast(tf.equal(pred,i),tf.int32))
        Prec.append(tp / (tp + fp))
        Rec.append(tp / (tp + fn))
        f1.append((2 * Prec[-1] * Rec[-1]) / (Prec[-1] + Rec[-1]))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, pred), tf.float32))
    
    tf.summary.scalar("Accuracy", accuracy)
    tf.summary.scalar("c1_Precision", Prec[1])
    tf.summary.scalar("c1_Recall", Rec[1])
    tf.summary.scalar("c1_F1_Score", f1[1])
    tf.summary.scalar("c2_Precision", Prec[2])
    tf.summary.scalar("c2_Recall", Rec[2])
    tf.summary.scalar("c2_F1_Score", f1[2])
    tf.summary.scalar("c3_Precision", Prec[3])
    tf.summary.scalar("c3_Recall", Rec[3])
    tf.summary.scalar("c3_F1_Score", f1[3])

    eval_list.append(('Acc. ', accuracy))
    eval_list.append(('xentropy', losses['xentropy']))
    eval_list.append(('weight_loss', losses['weight_loss']))
    Prec = tf.convert_to_tensor(Prec)
    Rec = tf.convert_to_tensor(Rec)
    f1 = tf.convert_to_tensor(f1)
    eval_list.append(('Overall Precision ', tf.reduce_mean(Prec)))
    eval_list.append(('Overall Recall', tf.reduce_mean(Rec)))
    eval_list.append(('Overall F1 score ', tf.reduce_mean(f1)))

    return eval_list