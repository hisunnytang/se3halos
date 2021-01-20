import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader

# adapt primarily from the Deep Unsupervised Learning Coursework
# https://sites.google.com/view/berkeley-cs294-158-sp20/home

class AffineTransform(pl.LightningModule):
  def __init__(self, nvar, dims, mask_type='A'):
    super(AffineTransform, self).__init__()
    self.mask = self.build_mask(nvar, dims, mask_type)
    self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
    self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    n_hidden = 2
    hidden_size=64
    self.mlp = MLP(dims, n_hidden, hidden_size, 2* dims)

  def build_mask(self, nvar, dims, mask_type):
    if mask_type == 'A':
      mask = np.zeros(dims)
      mask[:nvar] = 1
      return torch.tensor(mask.astype('float32'))
    elif mask_type == 'B':
      mask = np.zeros(dims)
      mask[-nvar:] = 1
      return torch.tensor(mask.astype('float32'))      

  def forward(self, x, reverse=False):
    batch_size, dims = x.shape
    mask = self.mask.repeat(batch_size,1).to(self.device)
    x_ = x*mask

    log_s, t = self.mlp(x_).split(dims, dim=1)
    log_s = self.scale* torch.tanh(log_s) + self.scale_shift

    t = t * (1.0 - mask)
    log_s = log_s * (1.0 - mask)

    if reverse:  # inverting the transformation
        x = (x - t) * torch.exp(-log_s)
    else:
        x = x * torch.exp(log_s) + t
    return x, log_s

# classic multi-layer perceptron
class MLP(pl.LightningModule):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class RealNVPTrainer(pl.LightningModule):
  def __init__(self, features_column=None, learning_rate=2.e-4, n_epochs= 1000):
    super(RealNVPTrainer, self).__init__()
    if features_column is None:
      # halo_dmvir_dt_inst
      # we have dropped this, because there is negative values...
      features_column  = ['halo_macc', 'halo_spin',
                          'halo_scale_factor_firstacc', 'halo_c_to_a', 'halo_mvir_firstacc',
                          'halo_scale_factor_last_mm', 'halo_scale_factor_mpeak', 'halo_m500c',
                          'halo_halfmass_scale_factor', 'halo_rvir', 'halo_vpeak', 'halo_mpeak',
                          'halo_m_pe_diemer', 'halo_m2500c', 'halo_mvir', 'halo_voff',
                          'halo_b_to_a', 'halo_m200b', 'halo_vacc', 'halo_scale_factor_lastacc',
                          'halo_m200c', 'halo_rs', 'halo_nfw_conc',
                          'halo_mvir_host_halo']

    self.features_column = features_column
    self.dims = len(features_column)
    self.train_loader, self.val_loader = self.create_dataloaders()
    self.log_interval = 100
    self.n_epochs = n_epochs
    self.learning_rate = learning_rate
    self.n_batches_in_epoch = len(self.train_loader)

    # initialize the model
    self.flow = RealNVP(self.dims)
    self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)


  def create_dataloaders(self, sample_size =100000, dataframe=df_halos, batch_size=1024):
    # should absorb the above code and wrap it here
    # features_column = ['halo_spin', 'halo_m500c', 'halo_rvir', 
    #                   'halo_mvir', 'halo_vacc', 'halo_vmax']


    self.scaler = StandardScaler()
    df_train = self.scaler.fit_transform( 
        np.log10(dataframe[self.features_column].sample(sample_size)+1e-6)).astype('float')

    X_train, X_test = train_test_split(df_train, random_state =42)
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(X_test,  batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


  def forward(self, x):
    x = self.flow(x)
    return x

  def training_step(self, batch, batch_idx):
    x = batch
    log_prob = self.flow.log_prob(x.float())
    loss = -torch.mean(log_prob) / self.dims
    self.log('loss', loss, prog_bar=True, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x = batch
    log_prob = self.flow.log_prob(x.float())
    loss = -torch.mean(log_prob) / self.dims
    self.log('val_loss', loss, on_epoch=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    return self.validation_step(batch, batch_idx)

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      return optimizer

if __name__ == "__main__":
  rnvp_train = RealNVPTrainer()
  train, val = rnvp_train.create_dataloaders()
  trainer = pl.Trainer(gpus=1, max_epochs=100, auto_lr_find=True, progress_bar_refresh_rate=50)
  trainer.fit(rnvp_train, train_dataloader = train, val_dataloaders= val)