import pandas as pd
import scipy.stats
import warnings
import itertools
from itertools import combinations
import numpy as np
try:
    from Coefficient import Coefficient
    from ANOVA import ANOVA
except(ModuleNotFoundError):
    from utils.Coefficient import Coefficient
    from utils.ANOVA import ANOVA
    


import warnings

class Bin(ANOVA, Coefficient):
    def __init__(self):
        self.numeric_target_column_minimums=None
        self.numeric_feature_col_thresholds=None

    # a helper function that bins columns
    def binner(self,x,num_bins):
        """
        used 2x in Bin().
        """
        mn,mx=x.min(),x.max()
        bins=np.linspace(mn-1e-10,mx+1e-10,num_bins+1,endpoint=True)
        return np.digitize(x.copy(), bins, right=False)
    
    #examine  relationships prior to binning
    def pre_bin_relationships(self,df,cat_num_alpha:float=0.05,num_num_corr:float=0.6,numeric_columns:list|None=None,categoric_columns:list|None=None,detect_pseudo_numeric:bool=False,categoric_target:str|list|None=None,numeric_target:str|list|None=None): 
        """
        prepares data for Bin().pair_column_headers()
        #it calls kruskal_wallis or spearman coefficient to evaluate relationships
        """
        cat_num_df=self.cat_num_column_kruskal_wallis_relationships(df, alpha=cat_num_alpha,keep_similar=False, numeric_columns=numeric_columns,categoric_columns=categoric_columns,detect_pseudo_numeric=detect_pseudo_numeric,categoric_target=categoric_target,numeric_target=numeric_target)
        num_num_df=self.num_num_column_spearman_coefficient_relationships(df, corr=num_num_corr,keep_correlated=True,self_detect=True,numeric_columns=numeric_columns,detect_pseudo_numeric=detect_pseudo_numeric,target=numeric_target)
        return num_num_df,cat_num_df
    
    def pair_column_headers(self,num_num_df,cat_num_df):
        """
        processes output from Bin().pre_bin_relationships()
        prepares data for Bin().determine_min_number_of_bins()
        if either arg is entered as None, it returns None for that arg
        """
        if cat_num_df is not None:
            cat_num_pairs=tuple((i.category,i.numeric) for i in cat_num_df[['category','numeric']].itertuples())
        if num_num_df is not None:
            num_num_pairs=tuple([(i.numeric_1,i.numeric_2) for i in num_num_df[['numeric_1','numeric_2']].itertuples()]+[(i.numeric_2,i.numeric_1)  for i in num_num_df[['numeric_1','numeric_2']].itertuples()])
        return num_num_pairs, cat_num_pairs


    def get_abs_coefficient_stat(self,data,method='spearman'):
        """
        a function used in Bin().determine_min_number_of_bins()
        and passed to Bin().find_min_bins()

        """
        xcol,ycol=data.columns[0],data.columns[1]
        return np.abs(data[ycol].corr(data[xcol],method=method))



    def find_min_bins(self, data, y_col, x_columns, test_func, threshold, direction_of_relationship,method:str='spearman'):
        """
        direction_of_relationship='lower'|'greater' indicates area where bins are still related to other columns
        returns: 
            cols_min_max_and_stat={
                            'column_i': 
                                    {'min_w_relationship':int or None,
                                    'max_w_no_relationship':int or None,
                                    'threshold_stat':np.float64() or None}, 
                            'column_i':...}
            global_min_for_these_columns:int
        accepts test_func as arg. it can be self(ANOVA).one_way_kruskal_wallis(self,two_col_cat_num_df) or self.get_abs_coefficient_stat(self,data)
        """
        mins_for_each_col=[]
        cols_min_max_and_stat={}
        for col in x_columns:
            lowest_possible_bins,highest_possible_bins=1,data.shape[0]
            low=lowest_possible_bins
            high=highest_possible_bins
            min_relation_max_no_relation_stat=[None,None,None]# this is what sets:  cols_min_max_and_stat[col]={'min_w_relationship':min_relation_max_no_relation_stat[0],'max_w_no_relationship':min_relation_max_no_relation_stat[1],'threshold_stat':min_relation_max_no_relation_stat[2]}
            while low<=high:
                mid=(low+high)//2
                data['binned']=self.binner(data[y_col],mid)
                if direction_of_relationship=='greater':
                    stat = test_func(data[[col,'binned']],method=method)
                else:
                    stat = test_func(data[[col,'binned']])
                
                #print(f"bin: {mid}      stat: {stat}      threshold: {threshold}")
                
                if direction_of_relationship=='lower':
                    if stat>=threshold:
                        min_relation_max_no_relation_stat[1]=mid
                        low=mid+1
                    else:
                        min_relation_max_no_relation_stat[0]=mid
                        min_relation_max_no_relation_stat[2]=stat
                        high=mid-1

                elif direction_of_relationship=='greater':
                    if stat<threshold:
                        min_relation_max_no_relation_stat[1]=mid
                        low=mid+1
                    else:
                        min_relation_max_no_relation_stat[0]=mid
                        min_relation_max_no_relation_stat[2]=stat
                        high=mid-1
            
            if min_relation_max_no_relation_stat[0]!=None:
                mins_for_each_col.append(min_relation_max_no_relation_stat[0])
            cols_min_max_and_stat[col]={'min_w_relationship':min_relation_max_no_relation_stat[0],'max_w_no_relationship':min_relation_max_no_relation_stat[1],'threshold_stat':min_relation_max_no_relation_stat[2]}
        global_min_for_these_columns=min(mins_for_each_col)
        return global_min_for_these_columns , cols_min_max_and_stat




    def determine_min_number_of_bins(self,dataframe:pd.DataFrame, num_num_pairs:list|tuple, cat_num_pairs:list|tuple, original_value_count_threashold:int=5, min_coeff:float=0.6, max_p_value:float=0.05,coeff_method:str='spearman'):
        """
        takes output of Bin().pair_column_headers() as input
        shares alphas with Bin().pre_bin_relationships()
        uses self.find_min_bins() and binner() internally
        outputs: 
        result metrics = {'target column i': 
                                min_bins_that_maintain_global_relationships,
                        'target column n':.................}
        x_col_thresholds = {'feature column i': 
                                {'xi column_threshold': (min num bins with relationship,  
                                                        max num bins with no relationship, 
                                                        np.float64(coeff or p_value w/relationship))},
                            'feature column n':..........}
        """
        data=dataframe.copy()
        #extract the columns that will be y-target
        cols_to_bin=list(set([i[1] for i in cat_num_pairs]+[i[1] for i in num_num_pairs]))  #use index 1 for both 
        #track minumum bins and max,no-relationship bins
        minimums={}
        y_relation_to_x_col_thresholds={}
        #iterate through target-numeric columns
        for col in cols_to_bin:
            #extract the columns that will be x-features
            x_cat_columns=tuple(set(i[0] for i in cat_num_pairs if i[1]==col))
            x_num_columns=tuple(set(i[0] for i in num_num_pairs if i[1]==col))

            # make calls to func min bins and metrics
            min_number_of_bins_numerical,min_number_of_bins_categorical = None, None
            if len(x_cat_columns)>0:
                min_number_of_bins_categorical,column_binning_metrics_categorical=self.find_min_bins( data, col, x_cat_columns, self.one_way_kruskal_wallis, max_p_value, direction_of_relationship='lower')
            if len(x_num_columns)>0:
                min_number_of_bins_numerical,column_binning_metrics_numerical=self.find_min_bins( data, col, x_num_columns, self.get_abs_coefficient_stat, min_coeff, direction_of_relationship='greater',method=coeff_method)
            if len(x_num_columns)<1 and len(x_cat_columns)<1:
                warnings.warn(f"For {col}, There are no potential solutions at these thresholds.", UserWarning)
                continue
            #update threshold metrics
            elif len(x_num_columns)>0 and len(x_cat_columns)>0:
                column_metrics=column_binning_metrics_categorical | column_binning_metrics_numerical
                y_relation_to_x_col_thresholds[col]=column_metrics
                #update the minimum bin that retains   ALL tested relationships
                if min_number_of_bins_categorical is None and min_number_of_bins_numerical is None:
                    minimum = None
                elif min_number_of_bins_categorical is None:
                    minimum = min_number_of_bins_numerical
                elif min_number_of_bins_numerical is None:
                    minimum = min_number_of_bins_categorical
                else:
                    minimum = max(min_number_of_bins_numerical,min_number_of_bins_categorical)
                minimums[col]=minimum
            elif len(x_num_columns)>0 and len(x_cat_columns)<1:
                y_relation_to_x_col_thresholds[col]=column_binning_metrics_numerical
                #update the minimum bin that retains   ALL tested relationships
                if min_number_of_bins_numerical is None:
                    minimum = None
                else:
                    minimum = min_number_of_bins_numerical
                minimums[col]=minimum
            elif len(x_num_columns)<1 and len(x_cat_columns)>0:
                y_relation_to_x_col_thresholds[col]=column_binning_metrics_categorical
                #update the minimum bin that retains   ALL tested relationships
                if min_number_of_bins_categorical is None:
                    minimum = None
                else:
                    minimum = min_number_of_bins_categorical
                minimums[col]=minimum
        return minimums, y_relation_to_x_col_thresholds



    #=============================================================================================================================================================
    #this needs edge case when xfeature columns is none and/or ytarget columns is none, presently, it attempts to detect datatypes
    #needs to have a default max num bins, such as 30%
    #=============================================================================================================================================================

    
    def relational_binner(self,df,max_cat_to_numeric_p=0.05,min_coeff=0.6,original_value_count_threashold:int=5,numeric_columns=None,categoric_columns=None,categoric_target:str|list|None=None,numeric_target:str|list|None=None, coeff_method:str='spearman' ):
        """
        if numeric_columns or categorical_columns are left as None, they will be infered which can result in missing, or incorrect comparisons
        after calling this function, min bins ==> self.numeric_result_metrics[column]['min_bins']
        where original_value_count_threashold=5 is the default threashold. columns with <=threashold unique values won't be considered

        calling this function will initialize object attributes:
        self.numeric_min_bins = {
                        'numeric column 1': 
                                {'min_bins': min number of bins that maintain significant relationship with all other columns considered,
                                'global categorical column stat key': min number of bins that maintain significant relationship,  
                                'global numerical column stat key': min number of bins that maintain significant relationship
                                }, 
                        'numeric column 2':..........}
        and
        self.numeric_binning_metrics = {'numeric column': 
                                {'xi column_threshold': (max num bins with no relationship,min num bins with relationship, np.float64(coeff or p_value w/relationship))},
                            'numeric column 2':..........}

        """
        num_num_df,cat_num_df=self.pre_bin_relationships(df,cat_num_alpha=max_cat_to_numeric_p,num_num_corr=min_coeff,numeric_columns=numeric_columns,categoric_columns=categoric_columns,categoric_target=categoric_target,numeric_target=numeric_target)
        num_num_pairs, cat_num_pairs=self.pair_column_headers(num_num_df,cat_num_df)
        self.numeric_target_column_minimums, self.numeric_feature_col_thresholds = self.determine_min_number_of_bins(df, num_num_pairs, cat_num_pairs, original_value_count_threashold, min_coeff=min_coeff, max_p_value=max_cat_to_numeric_p,coeff_method=coeff_method )
        return self.numeric_target_column_minimums
    #=============================================================================================================================================================
    #=============================================================================================================================================================
    