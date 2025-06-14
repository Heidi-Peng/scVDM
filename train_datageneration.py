import torch
from denoising_diffusion_pytorch import GaussianDiffusion_twocondition, Trainer_transformer,transformer
from torch.utils.data import  DataLoader
import numpy as np
import torch.nn.functional as F
from preprocess import Normalization
from datasets_df import get_metabolic_data_label_two
import pandas as pd
import os
import anndata
import scanpy as sc
from scipy import sparse
import sklearn.preprocessing as skp
from CVAE import Encoder,Decoder
from CVAE import VAE as v
from tqdm import tqdm
os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda:1"




if __name__ == '__main__':


    #pbmc
    adata=anndata.read("train_pbmc.h5ad")
    sc.pp.filter_cells(adata,min_genes=200)
    sc.pp.filter_genes(adata,min_cells=3)
    data=Normalization(adata)
    common=data.X.toarray()
    np.where(common,common,np.nan)
    print(common)
    mm=skp.MinMaxScaler()
    common_nor=mm.fit_transform(common)
    data.X=np.nan_to_num(common_nor,nan=np.nanmin(common_nor))
    print(common_nor)


    cell_type_key="cell_type"#"cell_label"#"species"
    condition_key="condition"#"condition"
    ctrl_key="control"#"Control"#'unst'
    stim_key="stimulated"#"Hpoly.Day10"#'LPS6'
    train_celllabel="none"

    #train
    data_all=data[~(data.obs[cell_type_key] == train_celllabel)]
    meta_file=data_all.to_df()
    label_df=data_all.obs['cell_type']
    condition_df=data_all.obs['condition']
    print(meta_file)
    print(label_df)
    print(condition_df)

    test_meta_file=data_all.to_df()
    test_label_df=data_all.obs['cell_type']
    test_condition_df=data_all.obs['condition']
    print(test_meta_file)
    print(test_label_df)
    print(test_condition_df)


    print(meta_file.shape)
    in_features=meta_file.shape[1]
    pre_transfer = None
    subject_dat = get_metabolic_data_label_two(meta_file,label_df,condition_df, sub_qc_split=False, use_log=False, pre_transfer=pre_transfer)
    datas = {'subject': subject_dat}
    train_data=datas['subject']
    dataset=train_data

    input_dim = in_features
    batch_size = 256
    learning_rate = 0.001
    hidden_size = 800
    latent_size = 100
    n=7

    if torch.cuda.is_available():
        device = torch.device('cuda:1')

    #CVAE
    encoder = Encoder(input_dim, hidden_size, latent_size,conditional=True,num_labels=n)
    decoder = Decoder(latent_size, hidden_size, input_dim,conditional=True,num_labels=n)
    vae = v(encoder, decoder).to(device)

    PARAMS_PATH='./results/cvae_pbmc-2.pt'
    
    all_sample_tensor=torch.empty((0, latent_size)).to(device)
    sample_gens=[]
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    
    #CVAE
    vae.load_state_dict(original_state)
    vae.eval()
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True)
    prog_iter = tqdm(dataloader, desc="Testing", leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(prog_iter):
            inputs=batch_data['sample_x']
            label_inputs=batch_data['sample_y']
            condition_inputs=batch_data['sample_condition']
            inputs=inputs.to(device)
            label_inputs=label_inputs.to(device)
            condition_inputs=condition_inputs.to(device)
            z, mean, log_var=vae.Encoder(inputs,label_inputs,condition_inputs)
            all_sample_tensor=torch.cat([all_sample_tensor,z],0)
 
    sample_gens=all_sample_tensor.cpu().numpy()
    print(sample_gens)
    mm2=skp.MaxAbsScaler()
    sample_gens=mm2.fit_transform(sample_gens)
    print("inputs")
    print(sample_gens)

    df_z_sample_gens=pd.DataFrame(sample_gens)
    z_subject_dat = get_metabolic_data_label_two(df_z_sample_gens,label_df,condition_df, sub_qc_split=False, use_log=False, pre_transfer=pre_transfer)
    z_datas = {'subject': z_subject_dat}
    z_train_data=z_datas['subject']
    z_dataset=z_train_data

    model=transformer(
        timesteps = 1000,
        self_condition = False,
        num_units = 100,
        d_model = 100*2,
        nhead = 5,
        dim_feedforward = 1024,
        activation = "gelu",
        verbose = False,
        num_celltype=7,
        num_condition=2,
    )
    model.to(device)


    diffusion = GaussianDiffusion_twocondition(
    model,
    num_features= latent_size,
    timesteps = 600,
    loss_type='l2',
    objective ='pred_noise',
    beta_schedule="linear",
    auto_normalize = False
    )
    diffusion.to(device)

    trainer = Trainer_transformer(
        diffusion,
        dataset = z_dataset,
        train_batch_size =512,
        train_lr = 1e-4,
        train_num_steps =100000,         # total training steps
        # gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
    )    
    # data=torch.load("/model-pbmc.pt")
    # trainer.model.load_state_dict(data['model'])
    # trainer.step=data['step']
    # trainer.opt.load_state_dict(data['opt'])
    # if trainer.accelerator.is_main_process:
    #     trainer.ema.load_state_dict(data["ema"]

    trainer.train()

    #test
    test_subject_dat = get_metabolic_data_label_two(test_meta_file,test_label_df,test_condition_df, sub_qc_split=False, use_log=False, pre_transfer=pre_transfer)
    test_datas = {'subject': test_subject_dat}
    test_data=test_datas['subject']
    test_dataset=test_data

    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, pin_memory = True)
    test_prog_iter = tqdm(test_dataloader, desc="Testing", leave=False)



    # after a lot of training
    batch_size=5000
    trainer.load(10)
    trainer.model.to(device).eval()
    
    zs=torch.empty((0, latent_size)).to(device)
    all_sample_seq=torch.empty((0,latent_size)).to(device)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_prog_iter):

            inputs=batch_data['sample_x']
            label_inputs=batch_data['sample_y']
            condition_inputs=batch_data['sample_condition']            
            inputs=inputs.to(device)
            label_inputs=label_inputs.to(device)
            condition_inputs=condition_inputs.to(device)
            if (batch_idx==0):
                labels=label_inputs
                conditions=condition_inputs
            else:    
                labels=torch.cat([labels,label_inputs],0)
                conditions=torch.cat([conditions,condition_inputs],0)
            z, mean, log_var=vae.Encoder(inputs,label_inputs,condition_inputs)
            zs=torch.cat([zs,z],0)
            sampled_seq=trainer.model.sample(inputs.shape[0],label_inputs,condition_inputs)
            all_sample_seq=torch.cat([all_sample_seq,sampled_seq],0)
    print("raw_sample")
    print(sampled_seq)


            
    latent_sample_gens=all_sample_seq.cpu().numpy()
    latent_sample_gens=mm2.inverse_transform(latent_sample_gens)
    latent_sample_gens_inverse_tensor=torch.tensor(latent_sample_gens,device=device)
    print("reverse")
    print(latent_sample_gens)


    with torch.no_grad():
        decode_sampled_seq=vae.Decoder(latent_sample_gens_inverse_tensor,labels,conditions)
    sample_gen=decode_sampled_seq.cpu().numpy()
    print("decoder")
    print(sample_gen)


    latent_sample_gens_np=latent_sample_gens
    labels=labels.cpu().numpy()
    conditions=conditions.cpu().numpy()
    sample_label={0:'CD4T', 1:'CD14+Mono', 2:'B', 3:'CD8T', 4:'NK', 5:'FCGR3A+Mono', 6:'Dendritic'}
    sample_con={0:'control', 1:'stimulated'}
    keys=[]
    key_conditions=[]
    key_labels=[]
    key_condition_labels=[]
    var_name=[]
    for i in range(100):
        var_name.append(i)
    for index,label in enumerate(labels):
        label_str=sample_label[label]
        condition=conditions[index]
        condition_str=sample_con[condition]
        key=label_str+"_gen"
        key_label=label_str
        key_condition=condition_str+"_gen"
        key_condition_label=condition_str
        keys.append(key)
        key_labels.append(key_label)
        key_conditions.append(key_condition)
        key_condition_labels.append(key_condition_label)

    pred_adata_latent = anndata.AnnData(latent_sample_gens_np, obs={"cell_type":keys,
                                        "condition":  key_conditions},
                                var={"var_names": var_name})
    
    print(pred_adata_latent.shape)


    stim_key = "stimulated"
    ctrl_key = "control"
    cell_type_key = "cell_type"
    condition_key="condition"

    train=data


    train_stim=train[train.obs["condition"] == 'stimulated']
    train_control=train[train.obs["condition"] == 'control']


    for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]

        ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                            obs={condition_key: [f"control"] * len(cell_type_ctrl_data),
                                                cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                            var={"var_names": cell_type_ctrl_data.var_names})
        
        if sparse.issparse(cell_type_data.X):
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.A
        else:
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                                obs={condition_key: [f"stimulated"] * len(real_stim),
                                                    cell_type_key: [cell_type] * len(real_stim)},
                                                var={"var_names": cell_type_data.var_names})
        if idx == 0:
            all_data = ctrl_adata.concatenate(real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, real_stim_adata)

    pred_adata = anndata.AnnData(sample_gen, obs={"cell_type": keys,
                                        "condition":key_conditions },
                                var={"var_names": train.var_names})

    print(pred_adata.shape)
    all_data=all_data.concatenate(pred_adata)


    all_data.write_h5ad(f"./pbmc_gen.h5ad")


