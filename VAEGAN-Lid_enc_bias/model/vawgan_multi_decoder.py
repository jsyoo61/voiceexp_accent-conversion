from tensorflow.contrib import slim
import tensorflow as tf
import json
import sys
sys.path.append('../')
from util.layers import GaussianLogDensity, GaussianKLD, \
    GaussianSampleLayer, lrelu

with open('architecture-vawgan-vcc2016.json') as f:
        arch = json.load(f)
is_training=True

def conv2d_layer(
    inputs,
    num_outputs,
    kernel_size,
    strides,
    padding = 'same',
    normalizer_fn = tf.contrib.layers.instance_norm,
    activation_fn = None,
    weights_initializer = None,
    scope = None):

    conv_layer = tf.contrib.layers.conv2d(
        inputs = inputs,
        num_outputs = num_outputs,
        kernel_size = kernel_size,
        data_format = 'NHWC',
        stride = strides,
        padding = padding,
        normalizer_fn = tf.contrib.layers.instance_norm,
        activation_fn = activation_fn,
        weights_initializer = weights_initializer,
        scope = scope)

    return conv_layer

def conv2d_traspose_layer(
    inputs,
    num_outputs,
    kernel_size,
    strides,
    padding = 'same',
    normalizer_fn = tf.contrib.layers.instance_norm,
    activation_fn = None,
    weights_initializer = None,
    scope = None):

    conv_layer = tf.contrib.layers.conv2d_transpose(
        inputs = inputs,
        num_outputs = num_outputs,
        kernel_size = kernel_size,
        data_format = 'NHWC',
        stride = strides,
        padding = padding,
        normalizer_fn = tf.contrib.layers.instance_norm,
        activation_fn = activation_fn,
        weights_initializer = weights_initializer,
        scope = scope)

    return conv_layer

def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, y = tf.nn.sigmoid(gates), name = name)

    return activation


def id_bias_add_2d(inputs, id):
    num_neuron = inputs.shape.dims[3].value
    id_reshaped = tf.reshape(id, [1, 1, 1, id.shape.dims[-1].value])
    bias = tf.layers.dense(inputs = id_reshaped, units = num_neuron)
    bias_reshaped = tf.reshape(bias, [1, 1, 1, tf.shape(inputs)[3]])
    bias_tiled = tf.tile(bias_reshaped , [1, tf.shape(inputs)[1], tf.shape(inputs)[2], 1])
    inputs_bias_added = inputs + bias_tiled

    return inputs_bias_added

def id_bias_add_2d_twice(inputs, id_1, id_2):

    bias_added_2d = id_bias_add_2d(inputs, id_1)
    bias_added_2d_twice = id_bias_add_2d(bias_added_2d, id_2)

    return bias_added_2d_twice

def encode(x,y,is_training,scope_name,mode='train'):
        x = tf.expand_dims(x, -1)
        with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
          x = id_bias_add_2d(x, y)

          E1 = conv2d_layer(x, 32, kernel_size = [3,9], strides=[1,1], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope = "E1_conv")
          E1_gated = conv2d_layer(x, 32, kernel_size = [3,9], strides=[1,1], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope ="E1_gated_conv")
          E1_GLU = gated_linear_layer(E1,E1_gated,"E1_GLU")
          E1_GLU = id_bias_add_2d(E1_GLU, y)

          E2 = conv2d_layer(E1_GLU,64, kernel_size = [4,8], strides=[2,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope ="E2_conv")
          E2_gated = conv2d_layer(E1_GLU, 64, kernel_size = [4,8], strides=[2,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope = "E2_gated_conv")
          E2_GLU = gated_linear_layer(E2,E2_gated,"E2_GLU")
          E2_GLU = id_bias_add_2d(E2_GLU, y)

          E3 = conv2d_layer(E2_GLU, 128, kernel_size = [4,8], strides=[2,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope = "E3_conv")
          E3_gated = conv2d_layer(E2_GLU, 128, kernel_size = [4,8], strides=[2,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope = "E3_gated_conv")
          E3_GLU = gated_linear_layer(E3,E3_gated,"E3_GLU")
          E3_GLU = id_bias_add_2d(E3_GLU, y)

          z_mu = conv2d_layer(E3_GLU, 5, kernel_size = [1,1], strides=[9,1], padding = 'same', normalizer_fn = None, activation_fn= None,weights_initializer = None,scope = "E6_z_mu")
          z_lv = conv2d_layer(E3_GLU, 5, kernel_size = [1,1], strides=[9,1], padding = 'same', normalizer_fn = None, activation_fn= None,weights_initializer = None,scope = "E6_z_lv")

          if mode =='test':
            return z_mu, z_lv
          else:
            return z_mu, z_lv



def decode(z, y,is_training, scope_name,mode='train',tanh=False):
        with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
          z = id_bias_add_2d(z, y)

          G1 = conv2d_traspose_layer(z, 128, kernel_size = [9,5], strides=[9,1], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope ="G1_conv")
          G1_gated = conv2d_traspose_layer(z, 128, kernel_size = [9,5], strides=[9,1], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope = "G1_gated_conv")
          G1_GLU = gated_linear_layer(G1,G1_gated,"G1_GLU")
          G1_GLU = id_bias_add_2d(G1_GLU, y)

          G2 = conv2d_traspose_layer(G1_GLU, 64, kernel_size = [4,8], strides=[2,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope = "G2_conv")
          G2_gated = conv2d_traspose_layer(G1_GLU, 64, kernel_size = [4,8], strides=[2,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope ="G2_gated_conv")
          G2_GLU = gated_linear_layer(G2,G2_gated,"G2_GLU")
          G2_GLU = id_bias_add_2d(G2_GLU, y)

          G3 = conv2d_traspose_layer(G2_GLU, 32, kernel_size = [4,8], strides=[2,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope = "G3_conv")
          G3_gated = conv2d_traspose_layer(G2_GLU, 32, kernel_size = [4,8], strides=[2,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope ="G3_gated_conv")
          G3_GLU = gated_linear_layer(G3,G3_gated,"G3_GLU")
          G3_GLU = id_bias_add_2d(G3_GLU, y)

          x = conv2d_traspose_layer(G3_GLU, 1, kernel_size = [3,9], strides=[1,1], padding = 'same', normalizer_fn = None, activation_fn= None,weights_initializer = None,scope = "G6_conv")
          out = tf.squeeze(x, axis=[-1], name='out_squeeze')
          return out

def discriminate(x,y,is_training,scope_name):
        x = tf.expand_dims(x, -1)

        with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
          D1 = conv2d_layer(x, 32, kernel_size = [3,9], strides=[1,1], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope ="D1_conv")
          D1_gated = conv2d_layer(x, 32, kernel_size = [3,9], strides=[1,1], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope ="D1_gated_conv")
          D1_GLU = gated_linear_layer(D1,D1_gated,"D1_GLU")
          D1_GLU = id_bias_add_2d(D1_GLU, y)

          D2 = conv2d_layer(D1_GLU, 32, kernel_size = [3,8], strides=[1,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope ="D2_conv")
          D2_gated = conv2d_layer(D1_GLU, 32, kernel_size = [3,8], strides=[1,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope ="D2_gated_conv")
          D2_GLU = gated_linear_layer(D2,D2_gated,"D2_GLU")
          D2_GLU = id_bias_add_2d(D2_GLU, y)

          D3 = conv2d_layer(D2_GLU, 32, kernel_size = [3,8], strides=[1,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope ="D3_conv")
          D3_gated = conv2d_layer(D2_GLU, 32, kernel_size = [3,8], strides=[1,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope ="D3_gated_conv")
          D3_GLU = gated_linear_layer(D3,D3_gated,"D3_GLU")
          D3_GLU = id_bias_add_2d(D3_GLU, y)

          D4 = conv2d_layer(D3_GLU, 32, kernel_size = [3,6], strides=[1,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope ="D4_conv")
          D4_gated = conv2d_layer(D3_GLU, 32, kernel_size = [3,6], strides=[1,2], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm,activation_fn = None,weights_initializer = None,scope ="D4_gated_conv")
          D4_GLU = gated_linear_layer(D4,D4_gated,"D4_GLU")
          D4_GLU = id_bias_add_2d(D4_GLU, y)

          D5 = conv2d_layer(D4_GLU, 32, kernel_size = [36,5], strides=[36,1], padding = 'same', normalizer_fn = tf.contrib.layers.instance_norm, activation_fn= None,weights_initializer = None,scope ="D5_conv")

          o1 = tf.layers.dense(inputs = D5, units = 1, activation = tf.nn.leaky_relu,name =  "o1_dense")
          return o1
