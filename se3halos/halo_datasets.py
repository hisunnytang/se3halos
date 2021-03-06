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

from sklearn.model_selection import train_test_split


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

    self.prepare_features()
  
  # def feature_normalization_stats(self, features, pre_norm: Dict = {}):
  #   """precompute the feature stats and transormation for normalization"""
  #   for feat_name, values in features.items():
  #     if feat_name in method:

  #     else:
  #       # do only a minmax

  def prepare_features(self, columns = ['halo_mvir', 'halo_vx', 'halo_vy', 'halo_vz']):

    # log transform halo mass
    logM = np.log10(self.df['halo_mvir'].values)
    vx, vy, vz = self.df['halo_vx'].values, self.df['halo_vy'].values, self.df['halo_vz'].values
    logMnorm, logM_mean, logM_std = self.normalize(logM)
    vxnorm, vxmean, vxstd = self.normalize(vx)
    vynorm, vymean, vystd = self.normalize(vy)
    vznorm, vzmean, vzstd = self.normalize(vz)

    self.df['logMnorm'] = logMnorm
    self.df['vxnorm'] = vxnorm
    self.df['vynorm'] = vynorm
    self.df['vznorm'] = vznorm

  def normalize(self, x):
    mean, std = x.mean(), x.std()
    return (x-mean)/std, mean, std


  def get_target(self, idx, normalize=True):
    target = self.targets[idx]
    if normalize:
      target = (target - self.mean) / self.std
    return target

  def __len__(self):
    return len(self.groupCounts)

class haloDataset_tng(Dataset):
  """
  load the halo dataset from dataframe to batch it as dgl.graph

  ...

  Parameters
  ----------
  df: pd.DataFrame
    attributes of the halo data, ordered by the group of halos
  groupCounts: np.ndarray
    number of halos in each group, with size equal to the number of graphs/ groups of haloes (N)
  groupIndx: np.ndarray
    the index between which the halo are considered a group, with size equal to the (N, 2)
  k: int
    maximum number of nearest neighbor for each halo
  position_columns: [str, str, str]
    the columns in df that corresponds to the position of each halo
  velocity_columns: [str, str, str]
    the columns in df that corresponds to the velocity of each halo
  spin_columns: [str, str, str]
    the columns in df that corresponds to the spin of each halo  
  mass_columns:

  """
  def __init__(self, 
               df,
               groupCounts,
               groupIndx, 
               k: int,
               
               # Input columns from DM-only information 
               position_columns: Tuple[str, str, str],
               velocity_columns: Tuple[str, str, str], 
               spin_columns: Tuple[str, str, str],
               mass_columns: Tuple[str,...],
              #  scalar_feature_columns: Union( Tuple[str], () ) = (),
               log_normalize_columns: Tuple[str,...],
               
               # List of baryons properties
               target_columns: Tuple[str,...], 
              #  log_normalize_target: Tuple[str,...],
               fully_connected: bool=False,
               floor: float=1.0e-20):
    self.df = df
    self.groupCounts = groupCounts
    self.groupIndx   = groupIndx

    self.k = k

    self.floor     = floor
    self.log_floor = np.log10(floor)
    self.targets = np.log10(self.df[target_columns].values + self.floor)
    # TODO: use the training stats unlike the other papers

    # log normalize the targets
    # self.mean = np.array([ np.mean(t[t>self.log_floor])  for t in self.targets.T])
    # self.std  = np.array([ np.std (t[t>self.log_floor])  for t in self.targets.T])
    # self.norm_target = self.targets.copy()
    # for i, t in enumerate(self.targets.T):
    #   self.norm_target[t > self.log_floor, i] = (t[t > self.log_floor] - self.mean[i]) / self.std[i]
    # self.norm_target = np.nan_to_num(self.norm_target, nan=self.log_floor)

    # log normalize the targets
    self.mean = np.array([ np.mean(t[t>self.log_floor])  for t in self.targets.T])
    self.std  = np.array([ np.std (t[t>self.log_floor])  for t in self.targets.T])
    self.norm_target = self.targets.copy()
    self.norm_target = np.nan_to_num(self.norm_target, nan=self.log_floor)
    for i, t in enumerate(self.norm_target.T):
      flag = t >  self.log_floor
      print(i, target_columns[i], self.norm_target[flag, i].min(), self.log_floor)
      tmp = (t[flag] - self.mean[i]) / self.std[i]
      self.norm_target[flag,  i] = tmp

      flag = t >  self.log_floor
      self.norm_target[~flag, i] = tmp.min() - 2.0*self.std[i]

      print(i, target_columns[i], t[flag].min(), self.log_floor)

    self.position_columns = position_columns
    self.velocity_columns = velocity_columns
    self.mass_columns     = mass_columns
    self.spin_columns     = spin_columns
    self.log_normalize_columns = log_normalize_columns
    self.feature_columns  = feature_columns
    input_columns = self.position_columns + self.velocity_columns +  self.mass_columns + self.spin_columns + self.feature_columns
    self.standard_columns = list(set(input_columns) - set(self.log_normalize_columns))
    
    self.norm_columns = []
    self.standardize_features(self.log_normalize_columns, log=True)
    self.standardize_features(self.standard_columns,      log=False)
    self.df = self.df.astype(np.float32)


  def standardize_targets(self, columns, log=True):
    features = np.log10(self.df[columns]) if log else self.df[columns]
    norm_features = (features - features.mean(axis=0))/ features.std(axis=0)
    norm_columns  = [f"{c}_norm" for c in columns]
    self.df[norm_columns] = norm_features
    self.norm_columns += norm_columns

  def standardize_features(self, columns, log=True):
    features = np.log10(self.df[columns]) if log else self.df[columns]
    norm_features = (features - features.mean(axis=0))/ features.std(axis=0)
    norm_columns  = [f"{c}_norm" for c in columns]
    self.df[norm_columns] = norm_features
    self.norm_columns += norm_columns


  def get_target(self, idx, normalize=True):
    target = self.targets[idx]
    if normalize:
      target = (target - self.mean) / self.std
    return target

  def __len__(self):
    return len(self.groupCounts)

  
  def __getitem__(self, idx):
    start, end = self.groupIndx[idx]
    # start, end = self.groupIndx[idx], self.groupIndx[idx+1]
    halo_group = self.df.iloc[start: end]

    halos_position = torch.from_numpy(halo_group[[f"{c}_norm" for c in self.position_columns]].values)
    halos_vel      = torch.from_numpy(halo_group[[f"{c}_norm" for c in self.velocity_columns]].values)
    halos_spins    = torch.from_numpy(halo_group[[f"{c}_norm" for c in self.spin_columns]].values)
    halos_mvir     = torch.from_numpy(halo_group[[f"{c}_norm" for c in self.mass_columns]].values)
    halos_features  = torch.from_numpy(halo_group[[f"{c}_norm" for c in self.feature_columns]].values)

    # instead of using KNN graph
    # which contains self edges.....
    # halos_knn = dgl.knn_graph(halos_position, self.k)
    halos_knn = self.get_NearestNeightborGraph(halos_position, min(self.k, end-start))

    u, v  = halos_knn.edges()
    halos_knn.edata['d'] = (halos_position[u] - halos_position[v]).clone().detach() #[num_atoms,3]

    halos_knn.ndata['mass']     = halos_mvir
    halos_knn.ndata['features'] = halos_features
    halos_knn.ndata['velocity'] = halos_vel
    halos_knn.ndata['spin']     = halos_spins
    halos_knn.ndata['x'] = halos_position - halos_position.mean(dim=0)

    # y = self.get_target(torch.arange(start, end), normalize=True)

    y = self.norm_target[start:end]

    return halos_knn, y

  def get_NearestNeightborGraph(self, position, k):
    # nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(position)
    # distances, indices = nbrs.kneighbors(position, include_self= False)

    # src = np.hstack([np.ones(self.k)*i for i in range(len(position))]).astype(int)
    # dst = np.hstack(indices[:,1:]).astype(int)

    # g = dgl.graph((src,dst))
    
    if k == 1:
      sp_mat = kneighbors_graph(X = position, n_neighbors= k, include_self=True )
      sp_mat.data[0] = 0
    elif k <= self.k:
      sp_mat = kneighbors_graph(X = position, n_neighbors= k-1, include_self=False )
    else:
      sp_mat = kneighbors_graph(X = position, n_neighbors= k, include_self=False )
    g = dgl.from_scipy(sp_mat)
    return g

  
  def __getitem__(self, idx):
    start, end = self.groupIndx[idx]
    halo_group = self.df.iloc[start: end]

    halos_position = torch.from_numpy(halo_group[['halo_x', 'halo_y', 'halo_z']].values)
    halos_vel  = torch.from_numpy(halo_group[['vxnorm', 'vynorm', 'vznorm']].values)
    halos_mvir = torch.from_numpy(halo_group[['logMnorm']].values)

    # instead of using KNN graph
    # which contains self edges.....
    # halos_knn = dgl.knn_graph(halos_position, self.k)
    halos_knn = self.get_NearestNeightborGraph(halos_position, self.k)

    u, v  = halos_knn.edges()
    halos_knn.edata['d'] = (halos_position[u] - halos_position[v]).clone().detach() #[num_atoms,3]

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


class haloDataset_RandomMask(haloDataset):
  """similar to haloDataset, but the target is a randomly mask halo attributes"""
  def __init__(self, *args, **kwargs):
    super(haloDataset_RandomMask, self).__init__(*args, **kwargs)

  def __getitem__(self, idx):
    start, end = self.groupIndx[idx]
    halo_group = self.df.iloc[start: end]

    p = halo_group[['halo_x', 'halo_y', 'halo_z']].values
    v = halo_group[['vxnorm', 'vynorm', 'vznorm']].values
    m = halo_group[['logMnorm']].values

    # get a masked index, but not the most massive ones
    halos, _ = m.shape
    masked_idx = np.random.randint(1, halos)
    # position/ velocity should be a relative quantity,
    # here we set the reference 'center' 
    # to be the most massive halo in the cluster
    y_relpos = p[masked_idx] - p[0]
    y_mass   = m[masked_idx]
    y_relvel = v[masked_idx] - v[0]
    y = {}
    y['1'] = np.vstack([y_relpos, y_relvel])
    y['0'] = y_mass

    # remove the masked row
    x_p = np.vstack([p[:masked_idx], p[masked_idx+1:]])
    x_v = np.vstack([v[:masked_idx], v[masked_idx+1:]])
    x_m = np.vstack([m[:masked_idx], m[masked_idx+1:]])

    halos_position = torch.from_numpy(x_p)
    halos_vel      = torch.from_numpy(x_v)
    halos_mvir     = torch.from_numpy(x_m)

    # instead of using KNN graph
    # which contains self edges.....
    # halos_knn = dgl.knn_graph(halos_position, self.k)
    halos_knn = self.get_NearestNeightborGraph(halos_position, self.k)

    u, v  = halos_knn.edges()
    halos_knn.edata['d'] = (halos_position[u] - halos_position[v]).clone().detach() #[num_atoms,3]

    halos_knn.ndata['mass']     = halos_mvir
    halos_knn.ndata['velocity'] = halos_vel
    halos_knn.ndata['x'] = halos_position - halos_position.mean(dim=0)

    #y = self.get_target(torch.arange(start, end), normalize=True)

    return halos_knn, y

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


def get_haloDataset(train_split = None,
                    fname= "/content/drive/MyDrive/halo_network/hlist_1.00035.list.halotools_v0p4.hdf5") -> Dataset:
  """read the halo catalog in as a pytorch Dataset"""
  df_halos = load_halos_catalog_df(fname = fname)
  grp_index, grp_count, grp_halos_df = order_halos_by_hostid(df_halos, min_halo_counts=5)
  if train_split is None:
    start_end_index_pair = list(zip(grp_index[:-1], grp_index[1:]))
    return haloDataset(df = grp_halos_df, groupCounts=grp_count, groupIndx=start_end_index_pair)

  train_idx, test_idx = train_test_split(range(len(grp_count)), train_size=train_split)

  train_start_end_pair = [[grp_index[t], grp_index[t+1]]  for t in train_idx]
  test_start_end_pair  = [[grp_index[t], grp_index[t+1]]  for t in test_idx]

  train_dataset = haloDataset(df = grp_halos_df, groupCounts=grp_count[train_idx], groupIndx=train_start_end_pair)
  test_dataset  = haloDataset(df = grp_halos_df, groupCounts=grp_count[test_idx], groupIndx=test_start_end_pair)

  return train_dataset, test_dataset

def get_MaskedHaloDataset(train_split = None,
                    min_halo_counts = 5,
                    fname= "/content/drive/MyDrive/halo_network/hlist_1.00035.list.halotools_v0p4.hdf5") -> Dataset:
  """read the halo catalog in as a pytorch Dataset"""
  df_halos = load_halos_catalog_df(fname = fname)
  grp_index, grp_count, grp_halos_df = order_halos_by_hostid(df_halos, min_halo_counts=min_halo_counts)
  if train_split is None:
    start_end_index_pair = list(zip(grp_index[:-1], grp_index[1:]))
    return haloDataset_RandomMask(df = grp_halos_df, groupCounts=grp_count, groupIndx=start_end_index_pair, k = min_halo_counts-1)

  train_idx, test_idx = train_test_split(range(len(grp_count)), train_size=train_split)

  train_start_end_pair = [[grp_index[t], grp_index[t+1]]  for t in train_idx]
  test_start_end_pair  = [[grp_index[t], grp_index[t+1]]  for t in test_idx]

  train_dataset = haloDataset_RandomMask(df = grp_halos_df, groupCounts=grp_count[train_idx], groupIndx=train_start_end_pair, k = min_halo_counts-1)
  test_dataset  = haloDataset_RandomMask(df = grp_halos_df, groupCounts=grp_count[test_idx], groupIndx=test_start_end_pair, k = min_halo_counts-1)

  return train_dataset, test_dataset

def visualize_halo_group(haloDS, index):
  graph, y = haloDS[index]
  print(graph)
  logmass = graph.ndata['mass'].clone().detach().numpy()
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