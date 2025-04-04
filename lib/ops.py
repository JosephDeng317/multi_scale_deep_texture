from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_slim as slim
from PIL import Image

def check_size(im):
    while(1):
        if (im.size[0] * im.size[1] > 3000000):
            w, h = int(0.9 * im.size[0]), int(0.9 * im.size[1])
            print('Image size [%i,%i] is too large, will be resized to [%i,%i]'\
                  %(im.size[0], im.size[1], w, h))
            im = im.resize((w, h), Image.BICUBIC) 
        else:
            return im

def gram(features):
    features = tf.reshape(features, [-1, features.shape[3]])
    return tf.matmul(features, features, transpose_a=True) / \
                 tf.cast(features.shape[0]*features.shape[1], dtype=tf.float32)

def total_variation_loss(image):
    tv_y_size = tf.size(image[:, 1:, :,], out_type=tf.float32)
    tv_x_size = tf.size(image[:, :, 1:,], out_type=tf.float32)
    tv_loss = (
                (tf.nn.l2_loss(image[:, 1:, :,] - image[:, :-1, :,]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:, :, 1:,] - image[:, :, :-1,]) /
                    tv_x_size)
              )
    return tv_loss

def get_layer_scope(name):
    target_layer = 'vgg_19/conv' + name[-2] + '/conv' + name[-2] + '_' + name[-1]  
    return target_layer  

def get_layer_list(layer, single_layer=False):
    style_layers = []
    if single_layer:
        if layer == 'VGG11':
            style_layers = ['VGG11']
        elif layer == 'VGG21':
            style_layers = ['VGG21']
        elif layer == 'VGG31':
            style_layers = ['VGG31']
        elif layer == 'VGG41':
            style_layers = ['VGG41']
        elif layer == 'VGG51':
            style_layers = ['VGG51']
        elif layer == 'VGG54':
            style_layers = ['VGG54']
        else:
            raise ValueError("NO THIS LAYER !")
    else:
        if layer == 'VGG11':
            style_layers = ['VGG11']
        elif layer == 'VGG21':
            style_layers = ['VGG11', 'VGG21']
        elif layer == 'VGG31':
            style_layers = ['VGG11', 'VGG21', 'VGG31']
        elif layer == 'VGG41':
            style_layers = ['VGG11', 'VGG21', 'VGG31', 'VGG41']
        elif layer == 'VGG51':
            style_layers = ['VGG11', 'VGG21', 'VGG31', 'VGG41', 'VGG51']
        elif layer == 'VGG54':
            style_layers = ['VGG11', 'VGG21', 'VGG31', 'VGG41', 'VGG51', 'VGG54']
        else:
            raise ValueError("No such layer in layer list.")
    return style_layers   

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    FLAGS = vars(FLAGS)
    for name, value in sorted(FLAGS.items()):
        if isinstance(value, float):
            print('\t%s: %f' % (name, value))
        elif isinstance(value, int):
            print('\t%s: %d' % (name, value))
        elif isinstance(value, str):
            print('\t%s: %s' % (name, value))
        elif isinstance(value, bool):
            print('\t%s: %s' % (name, value))
        else:
            print('\t%s: %s' % (name, value))
    print('End of configuration')

# VGG19 component
def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.
    Args:
        weight_decay: The l2 regularization coefficient.
    Returns:
        An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

# VGG19 net
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse=False,
           fc_conv_padding='VALID'):
    
    with tf.compat.v1.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return net, end_points

vgg_19.default_image_size = 224