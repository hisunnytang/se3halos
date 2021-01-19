
import sys

import numpy as np
import torch

from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber

from .halo_datasets import get_haloDataset, visualize_halo_group

class SE3Transformer(nn.Module):
    def __init__(self,  num_layers: int, atom_feature_size: int, num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
                 edge_dim: int=4, div: float=4, pooling: str='avg', n_heads: int=1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(2, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels)}

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        # FCblock = []
        # FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        # FCblock.append(nn.ReLU(inplace=True))
        # FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        FCblock = []
        FCblock.append(nn.Linear(64, 128))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(128, 1))


        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        logmass  = G.ndata['mass'].log10().unsqueeze(-1)
        velocity = G.ndata['velocity'].unsqueeze(1)
        h = {'0': logmass,
            '1': velocity}

        for layer in self.Gblock[:-1]:
            h = layer(h, G=G, r=r, basis=basis)
        h = h['0'].squeeze(-1) # n_halos x 64

        for layer in self.FCblock:
            h = layer(h)
        return h


if __name__ == '__main__':
    # pull the data
    haloDS = get_haloDataset()
    visualize_halo_group(haloDS, 86)

    # initialize model
    model_tfn = SE3Transformer(num_layers=4, 
                            atom_feature_size= 1,
                            num_channels=16,  
                            num_degrees=4, 
                            edge_dim=0,
                            div = 2,
                            pooling='max',
                            n_heads=2)
    
    # get 

    G   = haloDS[0][0]
    out = model_tfn(G)

    out_numnode, out_numfeat = out.shape
    assert out_numnode == G.num_nodes()
    assert out_numfeat == 1