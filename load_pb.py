import tensorflow as tf

model_file_name = './test_file_for_tf/models_pb/model.pb'
# test pb models which was transformed from ckpt files
#model_file_name = './test_file_for_tf/ckpt2pb/test_model.pb'

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
