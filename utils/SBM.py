import numpy as np

def SBM(p, q, N, n_classes=2):
    """Generate a SBM graph with n_classes.
    Args:
        p (flaot): the probability of connection between same cluster
        q ([type]): the probability of connection between different cluster
        N ([type]): the number of nodes
        n_classes (int, optional): the expexted number of clusters. Defaults to 2.
    """
    p_prime = 1 - np.sqrt(1-p)
    q_prime = 1 - np.sqrt(1-q)
    # ref to https://github.com/zhengdao-chen/GNN4CD/issues/5#issuecomment-806710492
    # for why using 1-sqrt(1-p)
    adj = np.ones((N, N)) * q_prime
    # TODO: handle the case when N is not divisible by n_classes
    cs = N // n_classes # cluster_size
    for i in range(n_classes-1):
        adj[i*cs : (i+1)*cs, i*cs : (i+1)*cs] = p_prime
    #adj[(n_classes-1)*cs:, (n_classes-1)*cs:] = p_prime
    adj = (np.random.rand(N, N) < adj).astype(int)
    adj = adj * (np.ones(N) - np.eye(N))
    adj = np.maximum(adj, adj.transpose())

    perm = np.random.permutation(N)

    labels = (perm // cs)
    #labels[labels >= n_classes] = n_classes-1

    adj = adj[perm][:, perm]

    return adj, labels