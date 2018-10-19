import tensorflow as tf

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model_ckpt/model.ckpt-3031.meta')
    

