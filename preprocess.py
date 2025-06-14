import numpy as np
import anndata
import scanpy as sc
import scanpy as sc
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
import sklearn.preprocessing as skp

def Normalization(adata, batch_key ="Batch",n_high_var = 2000,hvg_list=None, 
                     normalize_samples = True,target_sum=10000,log_normalize = True, 
                     normalize_features = False,scale_value=10.0,verbose=True,log=None):
    """
    Normalization of raw dataset 
    ------------------------------------------------------------------
    Argument:
        - adata: raw adata to be normalized

        - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.
    
        - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 1000, then the 1000 genes with the highest variance are designated as highly variable.
       
        - hvg_list: 'list',  a list of highly variable genes for seqRNA data
        
        - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.
        
        - target_sum: 'int',default 1e4,Total counts after cell normalization,you can choose 1e6 to do CPM normalization
            
        - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.
        
        - normalize_features: `bool`, If True, z-score normalize each gene's expression.

    Return:
        Normalized adata
    ------------------------------------------------------------------
    """
    
    n, p = adata.shape
    
    if(normalize_samples):
        sc.pp.normalize_total(adata, target_sum=target_sum)
        
    if(log_normalize):
        sc.pp.log1p(adata) 
    if normalize_features:
        if(len(adata.obs[batch_key].value_counts())==1): 
            sc.pp.scale(adata,max_value=scale_value)
            adata.obs["Batch"]=1
        else:
            adata_sep=[]
            for batch in np.unique(adata.obs[batch_key]):
                sep_batch=adata[adata.obs[batch_key]==batch]
                sc.pp.scale(sep_batch,max_value=scale_value)
                adata_sep.append(sep_batch)
            adata=sc.AnnData.concatenate(*adata_sep)
    
    return adata
  



        