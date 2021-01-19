
import dgl
import torch
from copy import deepcopy

from equivariant_attention.from_se3cnn.SO3 import rot

torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import DataLoader

from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import GSE3Res, GNormSE3, GConvSE3, GMaxPooling, get_basis_and_r
# The maximum feature type is harmonic degree 3
# from experiments.qm9.QM9 import QM9Dataset
from dgl.nn.pytorch.glob import AvgPooling


class GAvgVecPooling(nn.Module):
    """Graph Average Pooling module."""

    def __init__(self):
        super().__init__()
        self.pool = AvgPooling()

    def forward(self, features, G, **kwargs):
        print(f'before pool: {summary(features["1"])}')
        h_vec = []
        for i in range(3):
            h = features['1'][..., i]
            # print(f'before pool: {summary(h)}')
            h_vec.append(self.pool(G, h))
        return torch.cat(h_vec, axis=1)


def build_model():
    # The Fiber() object is a representation of the structure of the activations.
    # Its first argument is the number of degrees (0-based), so num_degrees=4 leads
    # to feature types 0,1,2,3. The second argument is the number of channels (aka
    # multiplicities) for each degree. It is possible to have a varying number of
    # channels/multiplicities per feature type and to use arbitrary feature types,
    # for this functionality check out fibers.py.
    num_degrees = 2
    num_features = 16  # todo added by Chen
    fiber_in = Fiber(1, num_features)
    fiber_mid = Fiber(num_degrees, 16)
    fiber_out = Fiber(2, 128)

    # We build a module from:
    # 1) a multihead attention block
    # 2) a nonlinearity
    # 3) a TFN layer (no attention)
    # 4) graph max pooling
    # 5) a fully connected layer -> 1 output

    model = nn.ModuleList([GSE3Res(fiber_in, fiber_mid),
                           # GNormSE3(fiber_mid),
                           # GConvSE3(fiber_mid, fiber_out, self_interaction=True),
                           # GConvSE3(fiber_out, Fiber(2, 1, structure=[(1, 1)]), self_interaction=True),
                           # GAvgVecPooling()
                           GConvSE3(fiber_mid, Fiber(2, 1, structure=[(1, 1)]), self_interaction=True),
                           ])
    fc_layer = nn.Linear(128, 1)
    return model


def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def summary(features):
    if isinstance(features, dict):
        for k, v in features.items():
            print(f'type: {k}; Size: {v.size()}')
    else:
        print(f'Size: {features.size()}')


def set_feat(G, R, num_features=16):
    G.edata['d'] = G.edata['d'] @ R
    # G.edata['w'] = torch.rand((G.edata['d'].size(0), 0))
    # G.ndata['x'] = torch.rand((G.ndata['x'].size(0), 0))
    G.ndata['f'] = torch.ones((G.ndata['f'].size(0), num_features, 1))
    print(G)

    # Run SE(3)-transformer layers: the activations are passed around as a dict,
    # the key given as the feature type (an integer in string form) and the value
    # represented as a Pytorch tensor in the DGL node feature representation.

    features = {'0': G.ndata['f']}
    return G, features


def apply_model(model, G, features, num_degrees=2):
    basis, r = get_basis_and_r(G, num_degrees - 1)
    for i, layer in enumerate(model):
        # print(f'feat before {layer}')
        # summary(features)
        features = layer(features, G=G, r=r, basis=basis)
        # print(f'feat after {layer}')
        # summary(features)
        # print('-' * 100)
        # print(i, features)
    # print(features)
    return features['1'][:, 0, :]
    # Run non-DGL layers: we can do this because GMaxPooling has converted features
    # from the DGL node feature representation to the standard Pytorch tensor rep.
    # print(features)
    # output = fc_layer(features)
    # print(output.size())