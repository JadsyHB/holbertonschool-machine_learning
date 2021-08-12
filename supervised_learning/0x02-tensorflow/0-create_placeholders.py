#!/usr/bin/env python3
"""
create placeholders module
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """returns 2 placeholders x and y"""
    x = tf.placeholder("float", name="x", shape=(None, nx))
    y = tf.placeholder("float", name="y", shape=(None, classes))
    return x, y
