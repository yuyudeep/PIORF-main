import numpy as np
import tensorflow as tf
import networkx as nx

import os
import datetime
import random
import warnings
import random
import json
import functools

from GraphRicciCurvature.OllivierRicci import OllivierRicci

def load_dataset(path, split):
    def _parse(proto, meta):
        feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}
        features = tf.io.parse_single_example(proto, feature_lists)
        out = {}
        for key, field in meta['features'].items():
            data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
            data = tf.reshape(data, field['shape'])
            if field['type'] == 'static':
                data = tf.tile(data, [meta['trajectory_length'], 1, 1])
                out[key] = data
            elif field['type'] == 'dynamic':
                out[key] = data
            elif field['type'] != 'dynamic':
                raise ValueError('invalid data format')
        return out


    """Load dataset."""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=1)

    ds = ds.prefetch(1)
    return ds


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


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_unique(senders, receivers):
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    return senders.numpy(), receivers.numpy()


def get_adj(N, senders, receivers):
    A = np.zeros(shape=(N, N))
    for i, j in zip(senders, receivers):
        if i != j:
            A[i, j] = 1.0
    return A


def pairwise_dist(A, B):
  
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)
    
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D


random.seed(0)
np.random.seed(0)
warnings.filterwarnings('ignore')

gpus = tf.config.list_physical_devices('GPU')
for i in range(len(gpus)):
	tf.config.experimental.set_memory_growth(gpus[i], True)


from digl import digl_rewire
from sdrf import sdrf_rewire
from fosr import fosr_rewire
from borf import OllivierRicciBORF

dataset_dir = '../data/cylinder_flow'


def rewire_baseline(dataset_dir, rewire_name):
    for mode in ['train', 'valid', 'test']:
        writer_dir = dataset_dir
        writer_path = os.path.join(writer_dir, f'{mode}_{rewire_name}.tfrecord')
        writer = tf.io.TFRecordWriter(writer_path)

        ds = load_dataset(dataset_dir, mode)
        counter = 0
        for ex in ds:
            example = {}
            for key, value in ex.items():
                value = value.numpy()
                if key == 'cells' or key == 'node_type' or key =='mesh_pos':
                    example[key] = np.expand_dims(value[0], 0)
                else:
                    example[key] = value

            pos = example['mesh_pos'][0]
            cells = example['cells'][0]
                
            N = pos.shape[0]
            senders, receivers  = triangles_to_edges(cells)
            edges = np.stack([senders, receivers]).astype(np.int64)
            A = get_adj(N, senders, receivers)
            G = nx.from_numpy_array(A)
            if rewire_name == 'digl':
                senders, receivers = digl_rewire(A, alpha=1/10, eps=4/100)
            elif rewire_name == 'sdrf':
                senders, receivers = sdrf_rewire(pos, edges, G, loops=10, remove_edges=False)
            elif rewire_name == 'fosr':
                senders, receivers = fosr_rewire(edges, num_iterations=20)
            elif rewire_name == 'borf':
                iters = 10
                batch_add=4
                batch_remove=2
                for _ in range(iters):
                    # Compute ORC
                    orc = OllivierRicciBORF(G, alpha=0)
                    orc.compute_ricci_curvature()
                    _C = sorted(orc.G.edges, key=lambda x: orc.G[x[0]][x[1]]['ricciCurvature']['rc_curvature'])

                    # Get top negative and positive curved edges
                    most_pos_edges = _C[-batch_remove:]
                    most_neg_edges = _C[:batch_add]

                    # Add edges
                    for (u, v) in most_neg_edges:
                        pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
                        p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
                        p, q = pi.index[p], pi.columns[q]
                        
                        if(p != q and not G.has_edge(p, q)):
                            G.add_edge(p, q)

                    # Remove edges
                    for (u, v) in most_pos_edges:
                        if(G.has_edge(u, v)):
                            G.remove_edge(u, v)

                edges = np.array(G.edges).astype(np.int32)
                senders = edges[:, 0]
                receivers = edges[:, 1]
                senders, receivers = np.concatenate([senders, receivers], axis=0), np.concatenate([receivers, senders], axis=0)
                senders, receivers = get_unique(senders, receivers)
            
            feature_dict = {
                f'senders_{rewire_name}'   : bytes_feature(senders.astype(np.int32).tobytes()),
                f'receivers_{rewire_name}' : bytes_feature(receivers.astype(np.int32).tobytes())
                }
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())
            
            counter += 1
            print(rewire_name, mode, counter)


# digl, sdrf, fosr, borf
rewire_baseline(dataset_dir, 'digl') 
rewire_baseline(dataset_dir, 'sdrf') 
rewire_baseline(dataset_dir, 'fosr') 
rewire_baseline(dataset_dir, 'borf') 

