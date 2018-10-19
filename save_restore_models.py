import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from ckpt2pb import ckpt2pb


### save the modes (ckpt)
"""
path = './models_ckpt/'
model_name = 'model.ckpt'
w1 = tf.Variable(tf.constant(2.0, shape=[1], name= 'w1_name'))
w2 = tf.Variable(tf.constant(3.0, shape=[1], name='w2_name' ))

x1 = tf.placeholder(tf.float32, name='x1_name')
x2 = tf.placeholder(tf.float32, name='x2_name')

y = tf.add(x1 * w1, x2 * w2, name='my_add')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print(x1)
    print(x2)
    print(y)
    print(sess.run(y, feed_dict={ x1 : 10, x2 : 5}))
    saver.save(sess, path + model_name)
"""


### load the ckpt model without redefine the graph

"""
path = './models_ckpt/'
saver = tf.train.import_meta_graph(path+'model.ckpt.meta')
graph = tf.get_default_graph()

x1 = graph.get_tensor_by_name('x1_name:0')
x2 = graph.get_tensor_by_name('x2_name:0')
y = graph.get_tensor_by_name('my_add:0')

with tf.Session() as sess:
    saver.restore(sess, path + 'model.ckpt')
    print(sess.run(y, feed_dict={x1 : 2, x2 : 3}))

"""

### save models to pb file
"""
path = './models_pb/'

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

with tf.gfile.GFile('./models_pb/model.pb', 'wb') as f:
    f.write(output_graph_def.SerializeToString())
"""

### load the pb model

model_file_name = './test/test_model.pb'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_graph_def = tf.GraphDef()
    with open(model_file_name, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def,name="")

    x1 = sess.graph.get_tensor_by_name('x1_name:0')
    x2 = sess.graph.get_tensor_by_name('x2_name:0')
    my_add = sess.graph.get_tensor_by_name('my_add:0')
    print(my_add)
    print(sess.run(my_add, feed_dict={x1:2,x2:3}))


# if __name__ == "__main__":
#     ckpt2pb('./models_ckpt/model.ckpt', './test_ckpt2pb/test_model.pb')

