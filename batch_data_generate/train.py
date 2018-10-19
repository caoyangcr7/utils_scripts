# this part is to train the data from the get_batch.py
# the code next is just a test
from get_batch import get_batch
import tensorflow as tf
# input_size means  the size of geo_map and score_map 
input_size = 512
# batch_size means the number of images in one train step
batch_size = 16

def main():
    with tf.Session() as sess:
        data_generator = get_batch(num_workers = 4, input_size=512, batch_size=16)
        max_train_steps = 1000
        for step in range(max_train_steps):
            every_batch_data = next(data_generator)
            model_loss = sess.run(model_loss, feed_dict={input_data: every_batch_data} )
            