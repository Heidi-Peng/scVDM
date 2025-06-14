import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from preprocess import Normalization
from datasets_df import get_metabolic_data_label_two
import anndata
import scanpy as sc
from scipy import stats
import sklearn.preprocessing as skp
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import time
from tqdm import tqdm
import numpy as np

def idx2onehot(idx,n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim,conditional,num_labels):
      super(Encoder, self).__init__()
      self.conditional=conditional
      if self.conditional:
          input_dim=input_dim+num_labels+2
      self.num_labels=num_labels
      self.FC_input = nn.Linear(input_dim, hidden_dim)
      self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
      self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
      self.FC_var   = nn.Linear (hidden_dim, latent_dim)
      self.leakyReLU = nn.LeakyReLU()
      

   
  def forward(self, x,c=None,c2=None):
      if self.conditional:
        c=idx2onehot(c,n=self.num_labels)
        c2=idx2onehot(c2,n=2)
        x=torch.cat((x,c),dim=-1)
        x=torch.cat((x,c2),dim=-1)
      h = self.leakyReLU(self.FC_input(x))
      h=self.leakyReLU(self.FC_input2(h))
      mean = self.FC_mean(h)
      log_var = self.FC_var(h)  
      sd=torch.exp(0.5*log_var)
      eps=torch.randn_like(sd)
      z=mean+sd*eps
      return z, mean, log_var
  


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim,conditional,num_labels):
        super(Decoder, self).__init__()
        self.conditional=conditional
        if self.conditional:
            input_size=latent_dim+num_labels+2
        else:
            input_size=latent_dim
        self.num_labels=num_labels
        self.FC_hidden = nn.Linear(input_size, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, output_dim)
        self.leakyReLU = nn.LeakyReLU()
    def forward(self, x,c=None,c2=None):
        if self.conditional:
            c=idx2onehot(c,n=self.num_labels)
            c2=idx2onehot(c2,n=2)
            x=torch.cat((x,c),dim=-1)
            x=torch.cat((x,c2),dim=-1)
        h=self.leakyReLU(self.FC_hidden(x))
        h=self.leakyReLU(self.FC_hidden2(h))
        mean=self.FC_mean(h)
        x_hat=self.ReLU(mean)
        return x_hat
        
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder,conditional=False,num_labels=0):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
                       
    def forward(self, x,c,c2):
        z, mean, log_var = self.Encoder(x,c,c2)
        x_hat=self.Decoder(z,c,c2)
        return x_hat, mean, log_var
    
    def predict(self,x_ctrl,x_stim,y_ctrl):
        z_latent_ctrl,_,_=self.Encoder(x_ctrl)
        z_latent_stim,_,_=self.Encoder(x_stim)
        latent_ctrl_avg=np.average(z_latent_ctrl,axis=0)
        latent_stim_avg=np.average(z_latent_stim,axis=0)
        delta=latent_stim_avg-latent_ctrl_avg
        
        predict_latent_ctrl=self.Encoder(y_ctrl)
        predict_latent_stim=predict_latent_ctrl+delta
        
        predict=self.Decoder(predict_latent_stim)
        return predict
        
        
def loss_function(x, x_hat,mean,log_var):
    MSE=0.5*torch.sum((x-x_hat).pow(2))
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    vae_loss=torch.mean(MSE+0.01*KLD)
    return vae_loss,MSE

def loss_function_encoder(mean,log_var):
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return KLD

    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count






if __name__ == '__main__':

    #pbmc
    adata=anndata.read("./train_pbmc.h5ad")
    sc.pp.filter_cells(adata,min_genes=200)
    sc.pp.filter_genes(adata,min_cells=3)
    data=Normalization(adata)
    print(data)
    common=data.X.toarray()
    np.where(common,common,np.nan)
    print(common)
    mm=skp.MinMaxScaler()
    common_nor=mm.fit_transform(common)
    print(common_nor)
    data.X=np.nan_to_num(common_nor,nan=np.nanmin(common_nor))
    print(common_nor)

    cell_type_key="cell_type"
    condition_key="condition"
    ctrl_key="control"
    stim_key="stimulated"
    train_celllabel="none"

    #train
    data_all=data[~(data.obs[cell_type_key] == train_celllabel)]

    #test
    # data_all=data[(data.obs[cell_type_key] == train_celllabel) & (data.obs[condition_key] == ctrl_key)]


    label_df=data_all.obs[cell_type_key]
    condition_df=data_all.obs[condition_key]
    meta_file=data_all.to_df()
    
    print(meta_file)
    print(label_df)
    print(condition_df)
    print(meta_file.shape)
    in_features=meta_file.shape[1]
    pre_transfer = None
    subject_dat = get_metabolic_data_label_two(meta_file,label_df,condition_df,sub_qc_split=False, use_log=False, pre_transfer=pre_transfer)
    datas = {'subject': subject_dat}
    train_data=datas['subject']
    dataset=train_data

    input_dim = in_features
    batch_size = 512
    num_epochs = 200
    learning_rate = 0.001
    hidden_size = 800
    latent_size = 100
    n=7

    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
    prog_iter = tqdm(dataloader, desc="Training", leave=False)


    encoder = Encoder(input_dim, hidden_size, latent_size,conditional=True,num_labels=n)
    decoder = Decoder(latent_size, hidden_size, input_dim,conditional=True,num_labels=n)
    vae = VAE(encoder, decoder,conditional=True,num_labels=n).to(device)

    loss_epoch = []
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss_sum=0.0
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        overall_loss = 0.0
        rec_loss = 0.0
        KLD_loss = 0.0

        for batch_idx, batch_data in enumerate(prog_iter):
            inputs=batch_data["sample_x"]
            label_inputs=batch_data['sample_y']
            condition_inputs=batch_data['sample_condition']
            inputs=inputs.to(device, dtype=torch.float32)
            label_inputs=label_inputs.to(device)
            condition_inputs=condition_inputs.to(device)
            x_hat, mean, log_var= vae(inputs,label_inputs,condition_inputs)
            loss,MSE,r_value= loss_function(inputs, x_hat,mean,log_var)
            overall_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(),batch_size)
            batch_time.update(time.time() - end)      
            end = time.time()

            if batch_idx % 20 == 0:
                print('Epoch: [{}][{}/{}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, batch_idx, len(dataloader), batch_time=batch_time,loss=losses))
                print("mse:",MSE)
        
        batch_size = len(dataloader)
        loss_epoch.append(overall_loss / batch_size)
        plt.figure()
        plt.plot(loss_epoch)
        plt.savefig('./figs/CVAE.png')


        if epoch > 50 and epoch % 100==0:
            milestone=epoch // 100
            all_state = {
            'original_state': vae.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'loss_epoch':loss_epoch
            }
            torch.save(all_state, str(f'./results/cvae_pbmc-{milestone}.pt'))
        print("overall_loss:"+str(overall_loss/(batch_idx*batch_size))+" rec_loss:"+str(rec_loss/(batch_idx*batch_size))+" KLD_loss:"+str(KLD_loss/(batch_idx*batch_size)))
