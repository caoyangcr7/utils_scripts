import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

### save models to pb file

path = './test_file_for_tf/models_pb/'

w1 = tf.Variable(tf.constant(2.0, shape=[1], name= 'w1_name'))
w2 = tf.Variable(tf.constant(3.0, shape=[1], name='w2_name' ))

x1 = tf.placeholder(tf.float32, name='x1_name')
x2 = tf.placeholder(tf.float32, name='x2_name')

y = tf.add(x1 * w1, x2 * w2, name='my_add')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['my_add'])

with tf.gfile.GFile('./test_file_for_tf/models_pb/model.pb', 'wb') as f:
    f.write(output_graph_def.SerializeToString())
