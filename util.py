import enum
import os
import json
import re

import tensorflow as tf


class NodeType(enum.IntEnum):
    NORMAL = 0
    DISABLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SYMMETRIC = 7
    SIZE = 9


def triangles_to_edges(faces, one_way=False):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    edges = tf.concat([faces[:, 0:2],
                        faces[:, 1:3],
                        tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = tf.reduce_min(edges, axis=1)
    senders = tf.reduce_max(edges, axis=1)
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
    # remove duplicates and unpack
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    # create two-way connectivity
    if one_way is True:
        return senders, receivers
    return tf.concat([senders, receivers], axis=0), tf.concat([receivers, senders], axis=0)


def get_meta(path):
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    return meta

def get_check_point_num(path):
    try:
        file = open(path, 'r')
        data = file.readlines()
        file.close()
        return re.sub(r'[^0-9]', '', data[0])
    except:
        return 0


def check_unique(senders, receivers):
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]
    return senders, receivers

def pairwise_dist(A, B):
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)
    
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D

