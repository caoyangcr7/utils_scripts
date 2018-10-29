import tensorflow as tf

### load the ckpt model without redefine the graph


path = './test_file_for_tf/models_ckpt/'
saver = tf.train.import_meta_graph(path+'model.ckpt.meta')
graph = tf.get_default_graph()

x1 = graph.get_tensor_by_name('x1_name:0')
x2 = graph.get_tensor_by_name('x2_name:0')
y = graph.get_tensor_by_name('my_add:0')

with tf.Session() as sess:
    saver.restore(sess, path + 'model.ckpt')
    print(sess.run(y, feed_dict={x1 : 2, x2 : 3}))


