import numpy as np

def train_data(hp, node_num, G, labeled_nodes):
    D = np.zeros((node_num, ))
    A = np.eye(node_num)
    N = len(labeled_nodes)
    node_id = [node[0] for node in labeled_nodes]
    train_nodes = node_id[:int(N*hp.ratio)]
    test_nodes = node_id[int(N*hp.ratio):]
    xs = np.array(train_nodes)
    ys_ = []
    for i in range(int(N*hp.ratio)):
        ys_.append(labeled_nodes[i][1])
    ys = np.array(ys_)
    unlabeled_nodes = list(set(G.nodes())-set(node_id))
    xu = np.array(test_nodes)
    yu_ = []
    for i in range(N-int(N*hp.ratio), N):
        yu_.append(labeled_nodes[i][1])
    yu = np.array(yu_)


    for edge in G.edges():
        A[int(edge[0])][int(edge[1])] += 1
        A[int(edge[1])][int(edge[0])] += 1
        D[int(edge[0])] += 1
        D[int(edge[1])] += 1
    D_ = np.diag(D**-0.5)
    L = np.matmul(np.matmul(D_, A), D_)
    return L, xs, ys, xu, yu
