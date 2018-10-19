from stn import spatial_transformer_network as transformer 
import numpy as np
import tensorflow as tf

# params
n_fc = 6
B, H, W, C = (2, 200, 200, 3)

# identity transform
initial = np.array([[1., 0, 0], [0, 1., 0]])
initial = initial.astype('float32').flatten()

# input placeholder
x = tf.placeholder(tf.float32, [B, H, W, C])

# localization network
W_fc1 = tf.Variable(tf.zeros([H*W*C, n_fc]), name='W_fc1')
b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
h_fc1 = tf.matmul(tf.zeros([B, H*W*C]), W_fc1) + b_fc1

# spatial transformer layer
h_trans = transformer(x, h_fc1)
