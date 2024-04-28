import os
import shutil
import numpy as np
import struct
import pickle
from scipy.sparse import coo_matrix

def load_flow(filename):
    # Flow is stored row-wise in order [channels, height, width].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    flow = None
    with open(filename, 'rb') as fin:
        width = struct.unpack('I', fin.read(4))[0]
        height = struct.unpack('I', fin.read(4))[0]
        channels = struct.unpack('I', fin.read(4))[0]
        n_elems = height * width * channels

        flow = struct.unpack('f' * n_elems, fin.read(n_elems * 4))
        flow = np.asarray(flow, dtype=np.float32).reshape([channels, height, width])

    return flow

def load_graph_info(filename, max_edges, max_nodes):

    assert os.path.isfile(filename), "File not found: {}".format(filename)

    with open(filename, 'rb') as fin:
        edge_total_size = struct.unpack('I', fin.read(4))[0]
        edges = struct.unpack('I' * (int(edge_total_size / 4)), fin.read(edge_total_size))
        edges = np.asarray(edges, dtype=np.int16).reshape(-1, 2).transpose()
        nodes_total_size = struct.unpack('I', fin.read(4))[0]
        nodes_ids = struct.unpack('I' * (int(nodes_total_size / 4)), fin.read(nodes_total_size))
        nodes_ids = np.asarray(nodes_ids, dtype=np.int32).reshape(-1)
        nodes_ids = np.sort(nodes_ids)

        edges_extent = np.zeros((2, max_edges), dtype=np.int16)
        edges_mask = np.zeros((max_edges), dtype=np.bool)
        edges_mask[:edges.shape[1]] = 1
        edges_extent[:, :edges.shape[1]] = edges

        nodes_extent = np.zeros((max_nodes), dtype=np.int32)
        nodes_mask = np.zeros((max_nodes), dtype=np.bool)
        nodes_mask[:nodes_ids.shape[0]] = 1
        nodes_extent[:nodes_ids.shape[0]] = nodes_ids

        fx = struct.unpack('f', fin.read(4))[0]
        fy = struct.unpack('f', fin.read(4))[0]
        ox = struct.unpack('f', fin.read(4))[0]
        oy = struct.unpack('f', fin.read(4))[0]
           
    return edges_extent, edges_mask, nodes_extent, nodes_mask, fx, fy, ox, oy

def load_adja_id_info(filename, src_mask, H, W, num_adja, num_neigb):

    assert os.path.isfile(filename), "File not found: {}".format(filename)
    assert num_adja<=8, "Num of adja is larger than 8"
    assert num_neigb<=8, "Num of neighb is larger than 8"
    src_v_id = np.zeros((H*W, num_adja), dtype=np.int16)
    src_neigb_id = np.zeros((H*W, num_neigb), dtype=np.int32)
    with open(filename, 'rb') as fin:
        neigb_id, value_id = pickle.load(fin)
        assert((src_mask.sum())==value_id.shape[0])

        for i in range(num_adja):
            src_v_id[src_mask.reshape(-1), i] = value_id[:, i]
        for i in range(num_neigb):
            src_neigb_id[src_mask.reshape(-1), i] = neigb_id[:, i]
    src_v_id = src_v_id.transpose().reshape(num_adja, H, W)
    src_neigb_id = src_neigb_id.transpose().reshape(num_neigb, H, W)

    return src_v_id, src_neigb_id

def save_flow(filename, flow_input):
    flow = np.copy(flow_input)

    # Flow is stored row-wise in order [channels, height, width].
    assert len(flow.shape) == 3
    
    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', flow.shape[2]))
        fout.write(struct.pack('I', flow.shape[1]))
        fout.write(struct.pack('I', flow.shape[0]))
        fout.write(struct.pack('={}f'.format(flow.size), *flow.flatten("C")))
