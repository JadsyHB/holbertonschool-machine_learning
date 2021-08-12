#!/usr/bin/env python3
"""
create placeholders module
"""


import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """returns the predicition in tensor form"""
    create_layer = __import__('1-create_layer').create_layer
    leng = len(layer_sizes)
    out = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, leng):
        out = create_layer(out, layer_sizes[i], activations[i])
    return out
