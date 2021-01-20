import torch
from torch.utils.data import DataLoader
import dgl
import numpy as np
import pytorch_lightning as pl
from model import SE3Transformer
import halo_datasets

def collate(samples):
  graphs, y = map(list, zip(*samples))
  batched_graph = dgl.batch(graphs)
  # print(batched_graph, y)
  return batched_graph, torch.from_numpy(np.vstack(y))

class SE3_HaloTransformer(pl.LightningModule):
  def __init__(self, learning_rate=1e-3):
    super(SE3_HaloTransformer, self).__init__()

    self.learning_rate = learning_rate
    # initialize model
    self.model = SE3Transformer(num_layers=2, 
                                atom_feature_size= 1,
                                num_channels=16,  
                                num_degrees=4, 
                                edge_dim=0,
                                div = 2,
                                pooling='max',
                                n_heads=4)
    self.loss_fn = torch.nn.MSELoss()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_nb):
    x, y = batch
    loss = self.loss_fn(self(x), y)
    self.log('train_loss', loss, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    pred = self(x)
    loss = self.loss_fn(pred, y)
    self.log('val_loss', loss, on_epoch=True)
    logmass = x.ndata['mass']
    return (logmass, pred, y)
  
  def validation_epoch_end(self, val_step_outputs):
    mass, predictions, ytrue = [], [], []
    f, ax = plt.subplots(1, 3, 
                         sharey=True, sharex=True,
                         figsize=(15,5))
    for x, pred, y in val_step_outputs:
      mass.append(x.detach().cpu().numpy().flatten()) 
      ytrue.append(y.detach().cpu().numpy().flatten())
      predictions.append(pred.detach().cpu().numpy().flatten())
    mass = np.hstack(mass)
    ytrue= np.hstack(ytrue)
    pred = np.hstack(predictions)

    ax[0].hexbin(mass, ytrue)
    ax[1].hexbin(mass, pred)
    # ax[2].scatter(mass, pred - ytrue)
    ax[2].scatter(ytrue, pred)
    ax[2].plot( [ytrue.min(), ytrue.max()] , [ytrue.min(), ytrue.max()])

    ax[0].set_title('true mass concentration plot')
    ax[0].set_xlabel('mass')
    ax[0].set_ylabel('conc')
    
    ax[1].set_title('pred mass concentration plot')
    ax[1].set_xlabel('mass')
    ax[1].set_ylabel('conc')
    
    ax[2].set_title('ytrue vs ypred')
    ax[1].set_xlabel('true concentration')
    ax[1].set_ylabel('predicted concentration')
    plt.show()

    del mass
    del ytrue
    del pred


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



if __name__ == '__main__':
  train_ds, test_ds = halo_datasets.get_haloDataset(train_split=0.8)

  train_loader = DataLoader(train_ds, 
                          batch_size=16, 
                          shuffle=True, 
                          collate_fn=collate,)

  test_loader  = DataLoader(test_ds, 
                            batch_size=16, 
                            shuffle=True, 
                            collate_fn=collate,)

  # Run learning rate finder
  pl_se3_transformer = SE3_HaloTransformer()
  trainer = pl.Trainer(gpus=1)
  lr_finder = trainer.tuner.lr_find(pl_se3_transformer,  
                                    min_lr=1e-5, 
                                    max_lr=1e-2, 
                                    num_training=100,
                                    train_dataloader = train_loader)

  # Results can be found in
  print(lr_finder.results)

  # Plot with
  fig = lr_finder.plot(suggest=True)
  fig.show()

  # Pick point based on plot, or get suggestion
  new_lr = lr_finder.suggestion()


  trainer = pl.Trainer(gpus=1, max_epochs= 100)
  pl_se3_transformer = SE3_HaloTransformer()
  # update hparams of the model
  # pl_se3_transformer.hparams.learning_rate = 
  trainer.fit(pl_se3_transformer, 
              train_dataloader= train_loader, 
              val_dataloaders= test_loader )
