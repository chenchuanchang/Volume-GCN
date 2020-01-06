# -*- coding: utf-8 -*-
#/usr/bin/python3
from module import Graphs
import numpy as np
import argparse
import pickle

# os.environ['CUDA_VISIBLE_DEVICES'] = ""

def parse_args():
    parser = argparse.ArgumentParser(description="GCN")
    # model
    parser.add_argument('--dim', type=int, default=256,
                        help='Number of dimensions. Default is 256  (0-255).')

    parser.add_argument('--dataset', default='datasets/jet_mixfrac_0051_supervoxels.gexf',
                        help='Name of dataset')
    parser.add_argument('--node_num', type=int, default=None,
                        help='Number of nodes.')
    args = parser.parse_args()
    return args

def main():
    hp = parse_args()
    G = Graphs(hp)
    node_num = len(G.nodes())
    labeled_nodes = [0 for i in range(100)]
    hp.node_num = node_num
    hp.labeled_node = len(labeled_nodes)

    D = np.zeros((node_num,))
    A = np.eye(node_num)
    for edge in G.edges():
        A[int(edge[0])][int(edge[1])] += 1
        A[int(edge[1])][int(edge[0])] += 1
        D[int(edge[0])] += 1
        D[int(edge[1])] += 1
    D_ = np.diag(D ** -0.5)
    L = np.matmul(np.matmul(D_, A), D_)

    emb = np.zeros((node_num, hp.dim))

    for i in range(node_num):
        for j in range(hp.dim):
            emb[i][j] = G.node[str(i)][str(j)]
    w = (np.random.randn(hp.dim, hp.dim) / np.sqrt(hp.dim/2)).astype('float32')
    emb = np.matmul(np.matmul(L, np.matmul(np.matmul(L, emb), w)), w)
    f = open('node_embedding.emb', 'wb')
    pickle.dump(emb, f)

    f = open('node_embedding.emb', 'rb')
    emb = pickle.load(f)

if __name__ == '__main__':
    main()