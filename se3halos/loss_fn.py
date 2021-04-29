import torch
import torch.nn as nn

class RegressionClassificationLoss(nn.Module):
  def __init__(self, weight, floor=-20):
    super(RegressionClassificationLoss, self).__init__()
    self.weight = weight
    self.floor  = floor
    self.class_lossfn = nn.BCEWithLogitsLoss(  pos_weight=  torch.Tensor([weight]) )
    self.reg_lossfn   = nn.MSELoss()
      
  def forward(self, ypred, ytrue,):
    ytrue_cat   = (ytrue > self.floor).long().float() #.squeeze()
    flag        = (ytrue > self.floor) # 
    # flag        = torch.logical_or((ypred [:,0] > 0.0), (ytrue > self.floor)) #
    class_err = self.class_lossfn(ypred [:,0], ytrue_cat)
    if flag.sum() > 0:
      reg_err   = self.weight* self.reg_lossfn(ypred[:,1][flag] , ytrue.squeeze()[flag])
      if flag.sum() != len(flag):
        reg_err  += self.reg_lossfn(ypred[:,1][~flag] , ytrue.squeeze()[~flag])
    else:
      reg_err = 0.0
    return  reg_err + class_err

class combined_loss_function(nn.Module):
  def __init__(self, 
               weights = [2049.16, 2049.16, 0.0079, 2.257, 23.96, 23.96, 23.522, 954.0],
               floors  = [-20, -20,-20, -20, -20, -20,-20]):
    super(combined_loss_function,self).__init__()

    self.bhmass_efn   = RegressionClassificationLoss(weight = weights[0], floor = floors[0])
    self.bhmdot_efn   = RegressionClassificationLoss(weight = weights[1], floor = floors[1])
    self.bfld_eftn    = RegressionClassificationLoss(weight = weights[2], floor = floors[2])

    self.sfr_efn       = RegressionClassificationLoss(weight = weights[3], floor = floors[3])
    self.gasmetal_efn  = RegressionClassificationLoss(weight = weights[4] , floor = floors[4])
    self.starmetal_efn = RegressionClassificationLoss(weight = weights[5], floor = floors[5])
    
    self.windmass_efn  = RegressionClassificationLoss(weight = weights[6], floor = floors[6])

  
  def forward(self, ypred, ytrue):
    ypred_tmp = ypred.reshape((-1, 7, 2))

    err1 = self.bhmass_efn   (ypred_tmp[:,0], ytrue[:,0])
    err2 = self.bhmdot_efn   (ypred_tmp[:,1], ytrue[:,1])
    err3 = self.bfld_eftn    (ypred_tmp[:,2], ytrue[:,2])
    err4 = self.sfr_efn      (ypred_tmp[:,3], ytrue[:,3])
    err5 = self.gasmetal_efn (ypred_tmp[:,4], ytrue[:,4])
    err6 = self.starmetal_efn(ypred_tmp[:,5], ytrue[:,5])
    
    err7 = self.windmass_efn (ypred_tmp[:,6], ytrue[:,6])
    return err1 + err2 + err3 + err4 + err5 + err6 + err7