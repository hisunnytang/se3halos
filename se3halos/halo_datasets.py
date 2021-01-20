"""Data Input PipeLine
"""

from halotools.sim_manager import DownloadManager
from halotools.sim_manager import sim_defaults
from halotools.sim_manager import CachedHaloCatalog
import numpy as np
import pandas as pd
import os
import sys

import dgl
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from scipy.constants import physical_constants
from torch.utils.data import Dataset
from typing import Dict, Tuple, List 
from sklearn.neighbors import KDTree, NearestNeighbors

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

class haloDataset(Dataset):
  def __init__(self, 
               df,
               groupCounts,
               groupIndx, 
               k: int = 5,
              #  feature_columns: List[List] = ['halo_mvir', 'halo_vx', 'halo_vy', 'halo_vz']
               target_columns: List = ['halo_nfw_conc'], 
               fully_connected: bool=False):
    self.df = df
    self.groupCounts = groupCounts
    self.groupIndx   = groupIndx

    self.k = k
    self.targets = np.log10(self.df[target_columns].values)
    # TODO: use the training stats unlike the other papers
    self.mean = np.mean(self.targets)
    self.std = np.std(self.targets)

  
  # def feature_normalization_stats(self, features, pre_norm: Dict = {}):
  #   """precompute the feature stats and transormation for normalization"""
  #   for feat_name, values in features.items():
  #     if feat_name in method:

  #     else:
  #       # do only a minmax



  def get_target(self, idx, normalize=True):
    target = self.targets[idx]
    if normalize:
      target = (target - self.mean) / self.std
    return target

  def __len__(self):
    return len(self.groupCounts)

  
  def __getitem__(self, idx):
    start, end = self.groupIndx[idx], self.groupIndx[idx+1]
    halo_group = self.df.iloc[start: end]

    halos_position = torch.from_numpy(halo_group[['halo_x', 'halo_y', 'halo_z']].values)
    halos_vel  = torch.from_numpy(halo_group[['halo_vx', 'halo_vy', 'halo_vz']].values)
    halos_mvir = torch.from_numpy(halo_group[['halo_mvir']].values)

    # instead of using KNN graph
    # which contains self edges.....
    # halos_knn = dgl.knn_graph(halos_position, self.k)
    halos_knn = self.get_NearestNeightborGraph(halos_position, self.k)

    u, v  = halos_knn.edges()
    halos_knn.edata['d'] = torch.tensor(halos_position[u] - halos_position[v]) #[num_atoms,3]

    halos_knn.ndata['mass']     = halos_mvir
    halos_knn.ndata['velocity'] = halos_vel
    halos_knn.ndata['x'] = halos_position - halos_position.mean(dim=0)

    y = self.get_target(torch.arange(start, end), normalize=True)

    return halos_knn, y

  def get_NearestNeightborGraph(self, position, k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(position)
    distances, indices = nbrs.kneighbors(position)

    src = np.hstack([np.ones(self.k)*i for i in range(len(position))]).astype(int)
    dst = np.hstack(indices[:,1:]).astype(int)

    g = dgl.graph((src,dst))
    return g

def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def load_halos_catalog_df(fname: str = None) -> pd.DataFrame:
  """load the predownload halocatalogs with halotools
  Args:
    fname: the downloaded filelocation
  Return:
    df_halos: pandas DataFrame with halo information
  
  Note:
    To predownload the halocatlog data, example script from https://halotools.readthedocs.io/en/latest/quickstart_and_tutorials/quickstart_guides/working_with_halotools_provided_catalogs.html
    >>> from halotools.sim_manager import DownloadManager
    >>> dman = DownloadManager()
    >>> dman.download_processed_halo_table('bolplanck', 'rockstar', 0.5) # doctest: +SKIP
  """

  if fname is None:
    raise FileNotFoundError( "please first download a halocatalog from halotools https://halotools.readthedocs.io/en/latest/quickstart_and_tutorials/quickstart_guides/working_with_halotools_provided_catalogs.html and specify filename fname" )

  halocat = CachedHaloCatalog(update_cached_fname=True,
                              fname = fname)

  df_halos = pd.DataFrame(halocat.halo_table.as_array())
  return df_halos


def order_halos_by_hostid(df_halos: pd.DataFrame, min_halo_counts: int = 5) -> (np.array, np.array, pd.DataFrame):
  """Reorder the halocatalogs in terms of the host halo ID

  Args:
    df_halos: halo catalog
    min_halo_counts: ignore groups of halos with members less than this number
  Return:
    grp_index:    start and end index of groups of halos (length of halos + 1)
    grp_count:    number of halos in each group
    grp_halos_df: df_halos reindex according to the grouping

  Example Usage:
    start, end  = grp_index[0], grp_index[1]
    halo_group1 = grp_halos_df.iloc[start:end]
  """
  group_index  = []
  group_hostid = []
  group_halocount = []

  for hostid, group in df_halos.groupby('halo_hostid'):
    if len(group) > min_halo_counts:
      group_index.append(group.index.values.astype(np.float))
      group_hostid.append(hostid)
      group_halocount.append(len(group))

  group_index = np.hstack(group_index).astype(int)

  tmp = df_halos.iloc[group_index].reset_index()
  grp_index = np.hstack([[0], np.cumsum(group_halocount) ])
  grp_count = np.array(group_halocount)
  grp_halos_df = tmp

  return grp_index, grp_count, grp_halos_df


def get_haloDataset(fname= "/content/drive/MyDrive/halo_network/hlist_1.00035.list.halotools_v0p4.hdf5") -> Dataset:
  """read the halo catalog in as a pytorch Dataset"""
  df_halos = load_halos_catalog_df(fname = fname)
  grp_index, grp_count, grp_halos_df = order_halos_by_hostid(df_halos, min_halo_counts=5)
  return haloDataset(df = grp_halos_df, groupCounts=grp_count, groupIndx=grp_index)

def visualize_halo_group(haloDS, index):
  graph, y = haloDS[index]
  print(graph)
  logmass = graph.ndata['mass'].log10().clone().detach().numpy()
  logmass_norm = (logmass - logmass.min()) / logmass.std()
  pos     = graph.ndata['x'].clone().detach().numpy()[:,:2]
  # y = np.log10(y+1e-10)
  y_minmax = (y - y.min()) / (y.max() - y.min())
  y_color  = cm.hot(y_minmax[:,0])

  graph = graph.to_networkx().to_undirected()

  f, ax = plt.subplots()
  ax= nx.draw(graph, 
              pos = pos,
              node_size = logmass_norm*100,
              node_color = y_color, ax=ax, 
              alpha=0.4)
  sm = plt.cm.ScalarMappable(cmap=cm.hot, 
                             norm=plt.Normalize(vmin = 0, vmax=1))
  plt.colorbar(sm,)