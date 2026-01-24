import pandas as pd
import numpy as np
import itertools
from itertools import combinations




class Coefficient:
    def __init__(self):
        self.one_way_pearson_coefficient_overview=None

    def target_features_corr_to_rest(self,data:pd.DataFrame,target:str|list,method:str='pearson'):
        """
        Where method can be of 'pearson', 'kendall', 'spearman'
        Iterates through all input dataframe columns and caculates the correlation with target column(s)
        returns an unsorted dataframe data[number_1, number_2, Correlation]
        """
        if type(target)==str:
            target=[target]
        cols=data.columns
        correlations=[]
        targs=[]
        y_vars=[]
        for col in cols:
            for targ in target:
                if col==targ:
                    continue
                corr=data[targ].corr(data[col],method=method)
                correlations.append(corr)
                targs.append(targ)
                y_vars.append(col)
        return pd.DataFrame({'numeric_1':targs,'numeric_2':y_vars,'Correlation':correlations}) 

    def test_all_num_num_pearson_coefficient(self,data,self_detect:bool=True,numeric_columns:list|None=None,pseudo_numeric_columns:list|None=None,detect_pseudo_numeric:bool=True,target:list|str|None=None):
        """
        Where pseudo_numeric can be numeric labeled categorical or ordinal data
        """
        columns=[]
        if numeric_columns is None and pseudo_numeric_columns is None: 
            columns=list(data.select_dtypes('number').columns)
        if numeric_columns is not None:
            columns+=numeric_columns
        if pseudo_numeric_columns is not None:
            columns+=pseudo_numeric_columns
        if self_detect==True:
            pseudo_num=[]
            if detect_pseudo_numeric==True:
                for col in list(set(list(data.select_dtypes('object').columns)+list(data.select_dtypes('category').columns))):
                    if pd.to_numeric(data[col], errors='coerce').notna().all():
                        pseudo_num.append(col)
            columns+=list(data.select_dtypes('number').columns)+pseudo_num
        columns=list(set(columns))
        num_data=pd.DataFrame(data.copy())
        if target is not None:
            if type(target)==str:
                target=[target]
            return self.target_features_corr_to_rest(num_data[columns],target,method='pearson')
        num_data=num_data[columns].corr(method='pearson').unstack().reset_index(drop=False).rename(columns={'level_0':'numeric_1','level_1':'numeric_2',0:'Correlation'})
        mask=num_data['numeric_1']!=num_data['numeric_2']
        num_data = num_data.loc[mask].reset_index(drop=True)
        map_base=pd.DataFrame(list(itertools.combinations(columns, 2)),columns=['numeric_1','numeric_2'])
        num_data=map_base.merge(num_data,how='inner',on=['numeric_1','numeric_2'])
        return num_data
    
    
    def num_num_column_pearson_coefficient_relationships(self,data, corr=0.6,keep_correlated:bool=True,self_detect:bool=True,numeric_columns:list|None=None,pseudo_numeric_columns:list|None=None,detect_pseudo_numeric:bool=True,target:list|str|None=None):
        """
        takes corr as a parameter 
        if reject is True, observations with correlations >= corr are returned.
        """
        num_data=self.test_all_num_num_pearson_coefficient(data,self_detect,numeric_columns,pseudo_numeric_columns,detect_pseudo_numeric,target)
        if keep_correlated==True:
            num_data=num_data.loc[np.abs(num_data['Correlation'])>=corr].reset_index(drop=True)
        else: num_data = num_data.loc[np.abs(num_data['Correlation'])<corr].reset_index(drop=True)
        return num_data
    

    def test_all_num_num_spearman_coefficient(self,data,self_detect:bool=True,numeric_columns:list|None=None,pseudo_numeric_columns:list|None=None,detect_pseudo_numeric:bool=True,target:list|str|None=None):
        """
        Where pseudo_numeric can be numeric labeled categorical or ordinal data
        """
        columns=[]
        if numeric_columns is None and pseudo_numeric_columns is None: 
            columns=list(data.select_dtypes('number').columns)
        if numeric_columns is not None:
            columns+=numeric_columns
        if pseudo_numeric_columns is not None:
            columns+=pseudo_numeric_columns
        if self_detect==True:
            pseudo_num=[]
            if detect_pseudo_numeric==True:
                for col in list(set(list(data.select_dtypes('object').columns)+list(data.select_dtypes('category').columns))):
                    if pd.to_numeric(data[col], errors='coerce').notna().all():
                        pseudo_num.append(col)
            columns+=list(data.select_dtypes('number').columns)+pseudo_num
        columns=list(set(columns))
        num_data=pd.DataFrame(data.copy())
        if target is not None:
            if type(target)==str:
                target=[target]
            return self.target_features_corr_to_rest(num_data[columns],target,method='spearman')
        num_data=num_data[columns].corr(method='spearman').unstack().reset_index(drop=False).rename(columns={'level_0':'numeric_1','level_1':'numeric_2',0:'Correlation'})
        mask=num_data['numeric_1']!=num_data['numeric_2']
        num_data = num_data.loc[mask].reset_index(drop=True)
        map_base=pd.DataFrame(list(itertools.combinations(columns, 2)),columns=['numeric_1','numeric_2'])
        num_data=map_base.merge(num_data,how='inner',on=['numeric_1','numeric_2'])
        return num_data


    def num_num_column_spearman_coefficient_relationships(self,data, corr=0.6,keep_correlated:bool=True,self_detect:bool=True,numeric_columns:list|None=None,pseudo_numeric_columns:list|None=None,detect_pseudo_numeric:bool=True,target:list|str|None=None):
        """
        takes corr as a parameter 
        """
        num_data=self.test_all_num_num_spearman_coefficient(data,self_detect,numeric_columns,pseudo_numeric_columns,detect_pseudo_numeric,target)
        if keep_correlated==True:
            num_data=num_data.loc[np.abs(num_data['Correlation'])>=corr].reset_index(drop=True)
        else: num_data = num_data.loc[np.abs(num_data['Correlation'])<corr].reset_index(drop=True)
        return num_data