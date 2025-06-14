import numpy as np
import pandas as pd
import torch.utils.data as data


class BaseData(data.Dataset):
    ''' Base Data Class '''
    def __init__(self, X_df,pre_transfer=None):
        '''
        X_df: samples x peakes, dataframe;
        Y_df:samples x 4, the colnames are injection.order, batch, group and
        class, group is the representation for CRC(1) and CE(0), class is the
        representation for Subject(1) and QCs(0), -1 represeents None.
        '''
        super(BaseData, self).__init__()
        self.x_df = X_df
        self.pre_transfer = pre_transfer
        if self.pre_transfer is not None:
            self.x_df = self.pre_transfer(self.x_df)

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, indx):
        sample_x = self.x_df.values[indx]
        return sample_x

    def transform(self, trans):
        ''' transform X and Y '''
        self.x_df, self.x_df = trans(self.x_df, self.x_df)
        return self

    @property
    def num_features(self):
        ''' the number of peaks '''
        return self.x_df.shape[1]

    #@property
    #def num_batch_labels(self):
    #    ''' the number of batches '''
    #    return len(self.y_df['batch'].unique())


class ConcatData(BaseData):
    ''' concatenate two BaseData objects '''
    def __init__(self, *datas):
        x_dfs = pd.concat([d.x_df for d in datas], axis=0)
        #y_dfs = pd.concat([d.y_df for d in datas], axis=0)
        super(ConcatData, self).__init__(x_dfs, None)


class BaseData_label_two(data.Dataset):
    ''' Base Data Class '''
    def __init__(self, X_df,label,condition,pre_transfer=None):
        '''
        X_df: samples x peakes, dataframe;
        Y_df：samples x 4, the colnames are injection.order, batch, group and
        class, group is the representation for CRC(1) and CE(0), class is the
        representation for Subject(1) and QCs(0), -1 represeents None.
        '''
        super(BaseData_label_two, self).__init__()
        self.x_df,self.label,self.condition= X_df,label,condition
        self.pre_transfer = pre_transfer
        if self.pre_transfer is not None:
            self.x_df = self.pre_transfer(self.x_df)
        

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, indx):
        sample_x = self.x_df.values[indx]
        self.label.replace({'CD4T':0, 'CD14+Mono':1, 'B':2, 'CD8T':3, 'NK':4, 'FCGR3A+Mono':5, 'Dendritic':6},inplace=True)
        #self.label.replace({'mouse':0, 'rabbit':1, 'pig':2, 'rat':3},inplace=True)
        #self.label.replace({'Endocrine':0, 'Enterocyte':1, 'Enterocyte.Progenitor':2, 'Goblet':3, 'Stem':4, 'TA':5,'TA.Early':6, 'Tuft':7},inplace=True)
        y=self.label.values[indx]
        self.condition.replace({'control':0, 'stimulated':1},inplace=True)
        #self.condition.replace({'unst':0, 'LPS6':1},inplace=True)#'Hpoly.Day10':1},inplace=True)
        #self.condition.replace({'Control':0, 'Hpoly.Day10':1},inplace=True)#'Hpoly.Day10':1},inplace=True)
        z=int(self.condition.values[indx])
        sample={'sample_x': sample_x, 'sample_y': y, 'sample_condition':z}
        return sample

    def transform(self, trans):
        ''' transform X and Y '''
        self.x_df, self.x_df = trans(self.x_df, self.x_df)
        return self

    @property
    def num_features(self):
        ''' the number of peaks '''
        return self.x_df.shape[1]


class BaseData_label_two_input(data.Dataset):
    ''' Base Data Class '''
    def __init__(self, X_df,input_df,label,condition,pre_transfer=None):
        '''
        X_df: samples x peakes, dataframe;
        Y_df：samples x 4, the colnames are injection.order, batch, group and
        class, group is the representation for CRC(1) and CE(0), class is the
        representation for Subject(1) and QCs(0), -1 represeents None.
        '''
        super(BaseData_label_two_input, self).__init__()
        self.x_df,self.input_df,self.label,self.condition= X_df,input_df,label,condition
        self.pre_transfer = pre_transfer
        if self.pre_transfer is not None:
            self.x_df = self.pre_transfer(self.x_df)
            self.input_df = self.pre_transfer(self.input_df)
        

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, indx):
        sample_x = self.x_df.values[indx]
        sample_input=self.input_df.values[indx]
        self.label.replace({'CD4T':0, 'CD14+Mono':1, 'B':2, 'CD8T':3, 'NK':4, 'FCGR3A+Mono':5, 'Dendritic':6},inplace=True)
        y=self.label.values[indx]
        self.condition.replace({'control':0, 'stimulated':1},inplace=True)
        z=self.condition.values[indx]
        sample={'sample_x': sample_x,'sample_input':sample_input, 'sample_y': y, 'sample_condition':z}
        return sample

    def transform(self, trans):
        ''' transform X and Y '''
        self.x_df, self.x_df = trans(self.x_df, self.x_df)
        return self

    @property
    def num_features(self):
        ''' the number of peaks '''
        return self.x_df.shape[1]


class BaseData_label_two_batch(data.Dataset):
    ''' Base Data Class '''
    def __init__(self, X_df,label,condition,pre_transfer=None):
        '''
        X_df: samples x peakes, dataframe;
        Y_df:samples x 4, the colnames are injection.order, batch, group and
        class, group is the representation for CRC(1) and CE(0), class is the
        representation for Subject(1) and QCs(0), -1 represeents None.
        '''
        super(BaseData_label_two_batch, self).__init__()
        self.x_df,self.label,self.condition= X_df,label,condition
        self.pre_transfer = pre_transfer
        if self.pre_transfer is not None:
            self.x_df = self.pre_transfer(self.x_df)
        

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, indx):
        sample_x = self.x_df.values[indx]
        #self.label.replace({'Group1':0, 'Group2':1, 'Group3':2, 'Group4':3, 'Group5':4, 'Group6':5, 'Group7':6},inplace=True)
        self.label.replace({'T-cell':0, 'NK':1, 'B-cell':2, 'Monocyte':3, 'Macrophage':4, 'Dendritic':5, 'Endothelial':6, 'Smooth-muscle':7, 'Neutrophil':8, 'Stromal':9, 'Epithelial':10},inplace=True)
        y=self.label.values[indx]

        #self.condition.replace({'Batch1':0, 'Batch2':1, 'Batch3':2, 'Batch4':3, 'Batch5':4, 'Batch6':5},inplace=True)
        self.condition.replace({'Batch1':0, 'Batch2':1},inplace=True)
        z=self.condition.values[indx]
        sample={'sample_x': sample_x, 'sample_y': y, 'sample_condition':z}
        return sample

    def transform(self, trans):
        ''' transform X and Y '''
        self.x_df, self.x_df = trans(self.x_df, self.x_df)
        return self

    @property
    def num_features(self):
        ''' the number of peaks '''
        return self.x_df.shape[1]


class BaseData_label_two_batch_input(data.Dataset):
    ''' Base Data Class '''
    def __init__(self, X_df,input_df,label,condition,pre_transfer=None):
        '''
        X_df: samples x peakes, dataframe;
        Y_df:samples x 4, the colnames are injection.order, batch, group and
        class, group is the representation for CRC(1) and CE(0), class is the
        representation for Subject(1) and QCs(0), -1 represeents None.
        '''
        super(BaseData_label_two_batch_input, self).__init__()
        self.x_df,self.input_df,self.label,self.condition= X_df,input_df,label,condition
        self.pre_transfer = pre_transfer
        if self.pre_transfer is not None:
            self.x_df = self.pre_transfer(self.x_df)
            self.input_df = self.pre_transfer(self.input_df)
        

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, indx):
        sample_x = self.x_df.values[indx]
        sample_input=self.input_df.values[indx]
        #self.label.replace({'Group1':0, 'Group2':1, 'Group3':2, 'Group4':3, 'Group5':4, 'Group6':5, 'Group7':6},inplace=True)
        self.label.replace({'T-cell':0, 'NK':1, 'B-cell':2, 'Monocyte':3, 'Macrophage':4, 'Dendritic':5, 'Endothelial':6, 'Smooth-muscle':7, 'Neutrophil':8, 'Stromal':9, 'Epithelial':10},inplace=True)
        y=self.label.values[indx]

        #self.condition.replace({'Batch1':0, 'Batch2':1, 'Batch3':2, 'Batch4':3, 'Batch5':4, 'Batch6':5},inplace=True)
        self.condition.replace({'Batch1':0, 'Batch2':1},inplace=True)


        z=int(self.condition.values[indx])
        sample={'sample_x': sample_x,'sample_input':sample_input, 'sample_y': y, 'sample_condition':z}
        return sample

    def transform(self, trans):
        ''' transform X and Y '''
        self.x_df, self.x_df = trans(self.x_df, self.x_df)
        return self

    @property
    def num_features(self):
        ''' the number of peaks '''
        return self.x_df.shape[1]


class BaseData_batch_input(data.Dataset):
    ''' Base Data Class '''
    def __init__(self, X_df,input_df,condition,pre_transfer=None):
        '''
        X_df: samples x peakes, dataframe;
        Y_df:samples x 4, the colnames are injection.order, batch, group and
        class, group is the representation for CRC(1) and CE(0), class is the
        representation for Subject(1) and QCs(0), -1 represeents None.
        '''
        super(BaseData_batch_input, self).__init__()
        self.x_df,self.input_df,self.condition= X_df,input_df,condition
        self.pre_transfer = pre_transfer
        if self.pre_transfer is not None:
            self.x_df = self.pre_transfer(self.x_df)
            self.input_df = self.pre_transfer(self.input_df)
        

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, indx):
        sample_x = self.x_df.values[indx]
        sample_input=self.input_df.values[indx]
        self.condition.replace({'tof2':0, 'tof4':1, 'tof5':2},inplace=True)
        z=self.condition.values[indx]
        sample={'sample_x': sample_x,'sample_input':sample_input, 'sample_condition':z}
        return sample

    def transform(self, trans):
        ''' transform X and Y '''
        self.x_df, self.x_df = trans(self.x_df, self.x_df)
        return self

    @property
    def num_features(self):
        ''' the number of peaks '''
        return self.x_df.shape[1]


def get_metabolic_data_label_two(
    x_file, label,condition,pre_transfer=None, sub_qc_split=True, use_log=False,
    use_batch=None, use_samples_size=None, random_seed=None
):
    '''
    Read metabolic data file and get dataframes
    metabolic data (x_file) example:
        name,mz,rt,QC1,A1,A2,A3,QC2,A4\n
        M64T32,64,32,1000,2000,3000,4000,5000,6000\n
        M65T33,65,33,10000,20000,30000,40000,50000,60000\n
        ...
    sample information data (y_file) example:
        sample.name,injection.order,batch,group,class\n
        QC1,1,1,QC,QC\n
        A1,2,1,0,Subject\n
        A2,3,1,1,Subject\n
        A3,4,1,1,Subject\n
        QC2,5,2,QC,QC\n
        A4,6,2,0,Subject\n
        A5,7,2,1,Subject\n
        A6,8,2,1,Subject\n
        ...
    '''
    meta_df=x_file


    if use_log:
        meta_df = meta_df.applymap(np.log)
    if pre_transfer is not None:
        meta_df = pre_transfer(meta_df)

    print (meta_df)

    return BaseData_label_two(meta_df,label,condition)


def get_metabolic_data_label_two_input(
    x_file,input_file,label,condition,pre_transfer=None, sub_qc_split=True, use_log=False,
    use_batch=None, use_samples_size=None, random_seed=None
):
    '''
    Read metabolic data file and get dataframes
    metabolic data (x_file) example:
        name,mz,rt,QC1,A1,A2,A3,QC2,A4\n
        M64T32,64,32,1000,2000,3000,4000,5000,6000\n
        M65T33,65,33,10000,20000,30000,40000,50000,60000\n
        ...
    sample information data (y_file) example:
        sample.name,injection.order,batch,group,class\n
        QC1,1,1,QC,QC\n
        A1,2,1,0,Subject\n
        A2,3,1,1,Subject\n
        A3,4,1,1,Subject\n
        QC2,5,2,QC,QC\n
        A4,6,2,0,Subject\n
        A5,7,2,1,Subject\n
        A6,8,2,1,Subject\n
        ...
    '''
    meta_df=x_file
    input_df=input_file


    if use_log:
        meta_df = meta_df.applymap(np.log)
    if pre_transfer is not None:
        meta_df = pre_transfer(meta_df)
    print (meta_df)
    print(input_df)

    return BaseData_label_two_input(meta_df,input_df,label,condition)

def get_metabolic_data_label_two_batch(
    x_file, label,condition,pre_transfer=None, sub_qc_split=True, use_log=False,
    use_batch=None, use_samples_size=None, random_seed=None
):
    '''
    Read metabolic data file and get dataframes
    metabolic data (x_file) example:
        name,mz,rt,QC1,A1,A2,A3,QC2,A4\n
        M64T32,64,32,1000,2000,3000,4000,5000,6000\n
        M65T33,65,33,10000,20000,30000,40000,50000,60000\n
        ...
    sample information data (y_file) example:
        sample.name,injection.order,batch,group,class\n
        QC1,1,1,QC,QC\n
        A1,2,1,0,Subject\n
        A2,3,1,1,Subject\n
        A3,4,1,1,Subject\n
        QC2,5,2,QC,QC\n
        A4,6,2,0,Subject\n
        A5,7,2,1,Subject\n
        A6,8,2,1,Subject\n
        ...
    '''
    meta_df=x_file


    if use_log:
        meta_df = meta_df.applymap(np.log)
    if pre_transfer is not None:
        meta_df = pre_transfer(meta_df)
    print (meta_df)

    return BaseData_label_two_batch(meta_df,label,condition)



def get_metabolic_data_label_two_batch_input(
    x_file,input_file,label,condition,pre_transfer=None, sub_qc_split=True, use_log=False,
    use_batch=None, use_samples_size=None, random_seed=None
):
    '''
    Read metabolic data file and get dataframes
    metabolic data (x_file) example:
        name,mz,rt,QC1,A1,A2,A3,QC2,A4\n
        M64T32,64,32,1000,2000,3000,4000,5000,6000\n
        M65T33,65,33,10000,20000,30000,40000,50000,60000\n
        ...
    sample information data (y_file) example:
        sample.name,injection.order,batch,group,class\n
        QC1,1,1,QC,QC\n
        A1,2,1,0,Subject\n
        A2,3,1,1,Subject\n
        A3,4,1,1,Subject\n
        QC2,5,2,QC,QC\n
        A4,6,2,0,Subject\n
        A5,7,2,1,Subject\n
        A6,8,2,1,Subject\n
        ...
    '''
    meta_df=x_file
    input_df=input_file


    if use_log:
        meta_df = meta_df.applymap(np.log)
    if pre_transfer is not None:
        meta_df = pre_transfer(meta_df)

    print (meta_df)
    print(input_df)

    return BaseData_label_two_batch_input(meta_df,input_df,label,condition)



