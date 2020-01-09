import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

def get_trmat(bs, rot, trs):
    """ Builds transform matrix to compute consistent bit loss.

    Args: 
        bs: Size of minibatch.
        rot: Maximum rotation degree.
        trs: Maximum translation ratio.

    Returns:
        3*3 transform matrix.
    
    """
    rot = np.random.randint(-rot, rot)
    trs = np.random.uniform(-trs, trs)
    hf = int(np.random.choice([-1, 1]))

    pi = torch.tensor(np.pi)
    cosR = torch.cos(rot * pi / 180.0)
    sinR = torch.sin(rot * pi / 180.0)

    rotmat = torch.zeros(bs, 3, 3)
    trsmat = torch.zeros(bs, 3, 3)
    hfmat = torch.zeros(bs, 3, 3)
    scmat = torch.zeros(bs, 3, 3)

    rotmat[:, 0, 0] = cosR
    rotmat[:, 0, 1] = -sinR
    rotmat[:, 1, 0] = sinR
    rotmat[:, 1, 1] = cosR
    rotmat[:, 2, 2] = 1.0

    trsmat[:, 0, 0] = 1.0
    trsmat[:, 0, 2] = trs
    trsmat[:, 1, 1] = 1.0
    trsmat[:, 1, 2] = trs
    trsmat[:, 2, 2] = 1.0

    hfmat[:, 0, 0] = hf
    hfmat[:, 1, 1] = 1.0
    hfmat[:, 2, 2] = 1.0

    mats = [trsmat, rotmat, hfmat]
    theta = mats[0]
    for matidx in range(1, len(mats)):
        theta = torch.matmul(theta, mats[matidx])
    theta = theta[:, :2, :]
    return theta

def set_input(bs, z_dim, b_dim, device):
    """ Makes input random variable consisting of a continuous part and a binary part of input random variable.

    Args:
        bs: Size of minibatch.
        z_dim: The length of a continuous part of input random variable.
        b_dim: The length of a binary part of input random variable.

    Returns:
        The input random variable, concatenation of z_dim continuous random noise and b_dim binary random noise.
    """
    z_input = torch.FloatTensor(bs, z_dim).uniform_(0, 1).to(device)
    b_input = torch.FloatTensor(bs, b_dim).uniform_(-1, 1).to(device)
    b_input = (b_input.sign()+1)/2
    zb_input = torch.cat([z_input, b_input], dim=1)
    return zb_input

def get_prec_topn(query_code, database_code, query_labels, database_labels, topn=1000):
    """ Computes precision@topn with given query and database codes.

    Args: 
        query_code: The binary codes of query images. Every element is -1 or 1.
        database_code: The binary codes of database images. Every element is -1 or 1.
        query_labels: One-hot labels of query images.
        database_labels: One-hot labels of database images.
        topn: The number of retrieved images.

    Returns:
        Precision@topn of given query and database codes.
    """
    num_query = query_labels.shape[0]
    num_database = database_labels.shape[0]

    mean_topn = 0.0

    for i in trange(num_query):
        S = (query_labels[i, :] @ database_labels.t() > 0).float()
        relevant_num = S.sum().item()
        if not relevant_num:
            continue

        hamming_dist = 0.5 * \
            (database_code.shape[1] - query_code[i, :] @ database_code.t())
        S = S[torch.argsort(hamming_dist)]
        prec_topn = S[:topn].sum().item() / topn
        mean_topn += prec_topn

    mean_topn = mean_topn / num_query

    return mean_topn

def bit_entropy(codes, reduction='mean'):
    """ Computes the entropy of each bit of the given code.

    Args:
        codes: The codes to compute entropy.
        reduction: The way to reduce the dimension of output. 'mean' and 'sum' are possible; the default is 'mean'.

    Returns:
        The entropy of bits of given code.
    """
    eps = 1e-40
    entropy = -(codes*codes.clamp(eps).log() + (1-codes)*(1-codes).clamp(eps).log())

    if reduction == 'sum':
        entropy_loss = entropy.sum()
    elif reduction == 'mean':
        entropy_loss = entropy.mean()
    else:
        print('specify the reduction')
    
    return entropy_loss
