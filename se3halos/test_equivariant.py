
import dgl
import torch
from copy import deepcopy

from equivariant_attention.from_se3cnn.SO3 import rot

torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import DataLoader
from halo_datasets import get_haloDataset
from model import SE3Transformer



def summary(features):
    if isinstance(features, dict):
        for k, v in features.items():
            print(f'type: {k}; Size: {v.size()}')
    else:
        print(f'Size: {features.size()}')


def rotate_feature(G, R, num_features=16):
  G.edata['d'] = G.edata['d']@R
  G.ndata['velocity'] = G.ndata['velocity']@R
  return G

def test_model_equivaraince(model, graph, rot_ang = (10,30,45)):
  g1 = deepcopy(graph)
  g2 = deepcopy(graph)

  R  = rot(*rot_ang)
  g1 = rotate_feature(g1, R)
  out1 = model(g1)
  out2 = model(g2)
  error = (1 - out2/ out1).abs().mean().detach().numpy()
  print(f'error: {error}')
  assert error < 1e-5, f'it is not equivariant for rotation {rot_ang} with error {error:.2e} > 1e-5'

if __name__ == '__main__':
  model_se3 = SE3Transformer(num_layers=4, 
                             atom_feature_size= 1,
                             num_channels=16,  
                             num_degrees=4, 
                             edge_dim=0,
                             div = 2,
                             pooling='max',
                             n_heads=2)
  
  model_se3.eval()
  halods = get_haloDataset()
  X_graph, y = halods[0]

  test_model_equivaraince(model_se3, X_graph, rot_ang = (10, 20, 30))




