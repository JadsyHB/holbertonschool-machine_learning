#!/usr/bin/env python3
"""trains a mini batch"""


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """trains a loaded NNM using mini batch gd"""
    m = X_train.shape[0]
    with tf.Ssession() as s:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(s, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        acc = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        for i in range(epochs + 1):
            train_cost = s.run(loss, feed_dict={
                x: X_train,
                y: Y_train
            })
            train_acc = s.run(acc, feed_dict={
                x: X_train,
                y: Y_train
            })
            valid_cost = s.run(loss, feed_dict={
                x: X_valid,
                y: Y_valid
            })
            valid_acc = s.run(acc, feed_dict={
                x: X_valid,
                y: Y_valid
            })
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))
            if i == epochs:
                break
            X, Y = shuffle_data(X_train, Y_train)
            if (m % batch_size) is 0:
                total = m // batch_size
            else:
                total = (m // batch_size) + 1
            # for j in range(total):
