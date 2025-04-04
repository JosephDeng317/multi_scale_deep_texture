from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
from PIL import Image
import numpy as np

# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        image_raw = Image.open(FLAGS.target_dir)
        if FLAGS.texture_shape == [-1,-1]:
            image_raw = check_size(image_raw)
        else:
            image_raw = image_raw.resize((FLAGS.texture_shape[0], FLAGS.texture_shape[1]))
        if image_raw.mode != 'RGB':
            image_raw = image_raw.convert('RGB')
        image_raw = np.asarray(image_raw) / 255.0
        targets = preprocess(image_raw)
        samples = np.expand_dims(targets, axis=0)
    return samples

def generator(FLAGS, target, init=None):
    if init is not None:
        var = tf.Variable(init + tf.compat.v1.random_normal(tf.shape(init), mean=0.0, stddev=FLAGS.stddev))
    else:
        if FLAGS.texture_shape == [-1,-1]:
            shape = [1, target.shape[1], target.shape[2], 3]
        else:
            shape = [1, FLAGS.texture_shape[0], FLAGS.texture_shape[1], 3]
        var = tf.compat.v1.get_variable('gen_img', shape=shape,
                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
                  dtype=tf.float32, trainable=True)
    return tf.tanh(var)

def Synthesis(initials, targets, upsampling, FLAGS):
    with tf.compat.v1.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):
        if initials is not None:
            w, h = int(initials.shape[1]), int(initials.shape[2])
            try:
                initials = tf.constant(initials)
            except:
                pass
            if upsampling:
                # Replace tf.image.resize_bicubic with tf.image.resize using BICUBIC method
                initials = tf.image.resize(initials, [2 * w, 2 * h], method=tf.image.ResizeMethod.BICUBIC)
        gen_output = generator(FLAGS, targets, initials)

    # Calculating the generator loss
    with tf.name_scope('generator_loss'):
        with tf.name_scope('tv_loss'):
            tv_loss = total_variation_loss(gen_output)
        with tf.name_scope('style_loss'):
            _, vgg_gen_output = vgg_19(gen_output, is_training=False, reuse=tf.compat.v1.AUTO_REUSE)
            _, vgg_tar_output = vgg_19(targets, is_training=False, reuse=tf.compat.v1.AUTO_REUSE)
            style_layer_list = get_layer_list(FLAGS.top_style_layer, False)
            sl = tf.zeros([], dtype=tf.float32)
            ratio_list = [100.0, 1.0, 0.1, 0.0001, 1.0, 100.0]
            for i in range(len(style_layer_list)):
                tar_layer = style_layer_list[i]
                target_layer = get_layer_scope(tar_layer)
                gen_feature = vgg_gen_output[target_layer]
                tar_feature = vgg_tar_output[target_layer]
                diff = tf.square(gram(gen_feature) - gram(tar_feature))
                sl = sl + tf.reduce_mean(tf.reduce_sum(diff, axis=0)) * ratio_list[i]
            style_loss = sl

        gen_loss = style_loss + FLAGS.W_tv * tv_loss
        gen_loss = 1e6 * gen_loss

    with tf.name_scope('generator_train'):
        gen_tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        # Replace the ScipyOptimizerInterface with AdamOptimizer in TF2
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(gen_loss, var_list=gen_tvars)

    # Get list of VGG variables and create saver for restoring
    vgg_var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.compat.v1.train.Saver(vgg_var_list)

    # Start the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    def print_loss(gl, sl_val, tvl):
        if FLAGS.print_loss:
            print('gen_loss : %s' % gl)
            print('style_loss : %s' % sl_val)
            print('tv_loss : %s' % tvl)

    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init_op)
        vgg_restore.restore(sess, FLAGS.vgg_ckpt)
        print('Under Synthesizing ...')

        # Run training loop for max_iter steps
        for i in range(FLAGS.max_iter):
            gl, sl_val, tvl, _ = sess.run([gen_loss, style_loss, tv_loss, train_op])
            print_loss(gl, sl_val, tvl)

        gen_out, style_loss_out = sess.run([gen_output, style_loss])
        return gen_out, style_loss_out