import tensorflow as tf

### save the modes (ckpt)

path = './test_file_for_tf/models_ckpt/'
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

