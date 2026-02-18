import pandas as pd
import scipy.stats
import warnings
import itertools
from itertools import combinations
import numpy as np

#these use f.survival_function or t.survival_function) to calculate the p value
# scipy.stats.f.sf(f_score,dfn,dfd) where dfn is the numerator--mean_square(ie variance) and dfd is denominator-->error# this returns a p value for the f stat
#     #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
#     #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.f


class ANOVA:
    def __init(self):
        self.two_way_interaction_columns=None
        self.two_way_interaction_sizes=None
        self.one_way_df_ANOVA_overview=None
        self.one_way_kruskal_wallis_df_overview=None

    # =========================================================================================================================================================================

    # ========================================================================================================================================================================= 

    # PREPROCESS
    # for uniform 2-way-ANOVA interaction sizes
    # presently stratification is not supported, stochastic under/over sampling is. 
    # In future versions, stratifications can be achieved with pd.sample(weights=) fitted based on bins&Probabilities
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html 
    
    def create_uniform_interactions(self, xx_y_prep_df,min_size=5, override_min=False, ntile=None):
        """
        override_min will sample with replacement from small datasets
        if ntile is not none, the interseciton size will be the ntile of sizes that are greater than min, otherwise smallest acceptable interaciton's size will be used

        """
        three_col_XX_y=pd.DataFrame(xx_y_prep_df.copy())
        cols=three_col_XX_y.columns
        x1 = cols[0]
        x2 = cols[1]
        y = cols[2]

        #unique values for interactions
        x1_vars=three_col_XX_y[x1].unique()
        x2_vars=three_col_XX_y[x2].unique()

        #record interactions and sizes
        grouped=three_col_XX_y.groupby([x1,x2],as_index=False,observed=False)[y].agg('size')
        sufficient=grouped.loc[(grouped['size']>=min_size)]
        sizes=sufficient['size'].to_list()
        interactions=list(sufficient[[x1,x2]].itertuples(index=False,name=None))
        too_small_interactions=list(grouped.loc[(grouped['size']>0)&(grouped['size']<min_size)][[x1,x2]].itertuples(index=False,name=None))
        non_interactions=list(grouped.loc[(grouped['size']<=0)][[x1,x2]].itertuples(index=False,name=None))

        # concatinate a new dataframe with uniform interaction sizes
        result=[]
        if ntile is not None:
            n_rows=np.percentile(sizes,[ntile])[0].astype(int)
            #
            #warning
            for interaction in interactions:
                data=three_col_XX_y.loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].sample(n_rows,replace=False)
                result.append(data)
            if override_min==True:
                #
                #warning
                for interaction in too_small_interactions:
                    data=three_col_XX_y.loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].sample(n_rows,replace=False)
                    result.append(data)
        elif ntile is None:
            n_rows=min(sizes)
            for interaction in interactions:
                data=three_col_XX_y.loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].sample(n_rows)
                result.append(data)
            if override_min==True:
                #
                #warning
                for interaction in too_small_interactions:
                    data=three_col_XX_y.loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].sample(n_rows,replace=True)
                    result.append(data)
        if override_min==True:
            interactions=interactions+too_small_interactions
            too_small_interactions=[]
        
        #metrics
        self.two_way_interaction_sizes=grouped
        #over_under_samplimng_interaciton_size=n_rows # not stratified, but random sampling
        self.two_way_interaction_columns={'interactions':interactions,'insufficient':too_small_interactions,'non_interactive': non_interactions }
        return pd.concat(result)
       
    # =========================================================================================================================================================================
    #following is a 2-way ANOVA; future: implement Aligned Rank Transform (ART) and Permutation (randomization) ANOVA for non parametric
    # =========================================================================================================================================================================     
            
    # 2 way ANOVA
    # uses typ2 1 sum or sqares,, the statsmodels implementation: two_way_ANOVA_for_un_balanced_data() adjusts for type 1,2,or 3 and is not optimized for GPU

    def two_way_ANOVA(self,three_col_XX_y,unbalanced_interaction_sizes=True,verbose=True):
        """
        Where col in positions [:2] are catigorical and [2] is numeric
        data is preprocessed to have uniform observations counts across interaction
        """
        three_col_XX_y=pd.DataFrame(three_col_XX_y.copy())  #ensure CuDF if applicable. A check would be better, but this prototypes well
        cols=three_col_XX_y.columns
        a = cols[0]
        b = cols[1]
        y = cols[2]

        #create an interaction dataframe (tempararily keep non interaction groups)
        interaction = three_col_XX_y.groupby([a,b],as_index=False,observed=False)[y].agg(['mean','size'])

        # track interacting and non interacting factors
        interaction_factors=list(interaction.loc[interaction['size']>=5][[a,b]].itertuples(index=False,name=None))
        non_interacting_factors=list(interaction.loc[interaction['size']<5][[a,b]].itertuples(index=False,name=None))
        if verbose:
            print(f"Interacting factors: {interaction_factors}\nNon interacting: where count < 5 interaction factors: {non_interacting_factors}")

        # filter out non interaction factor groups
        interaction=interaction.loc[interaction['size']>0]

        if not unbalanced_interaction_sizes:
            nij=interaction['size'].mean()# this is okay because this is not a wieghted anova

        if not unbalanced_interaction_sizes:
            pass
            #assert ... f""
        if unbalanced_interaction_sizes==True:
            pass
            #assert

        #number of groups in a and in b
        a_number_of_groups = three_col_XX_y[a].nunique()
        b_number_of_groups = three_col_XX_y[b].nunique()

        #multiply num of groups by num_interact_ai_and_bj for groups in a & b
        if not unbalanced_interaction_sizes: 
            nij_multiply_a = nij*a_number_of_groups
            nif_multiply_b = nij*b_number_of_groups
        elif unbalanced_interaction_sizes==True:
            interaction['nij_product_a']=interaction['size']*a_number_of_groups
            interaction['nij_product_b']=interaction['size']*b_number_of_groups

        #grand mean
        overall_mean = three_col_XX_y[y].mean()

        #calculate sums of squares for t and e
        SST=((three_col_XX_y[y] - overall_mean)**2).sum()
        ### while broadcast subraction is unsuported, this merge is used as a middle step
        SSE_dataframe = three_col_XX_y.merge(interaction[[a,b,'mean']],on=[a,b],how='left',validate='m:1')
        SSE = ((SSE_dataframe[y]-SSE_dataframe['mean'])**2).sum()
        del SSE_dataframe

        #calculate sums of squares for groups of factors a & b
        if not unbalanced_interaction_sizes:
            mean_y_ai = three_col_XX_y.groupby(a,observed=True)[y].mean()#--------------------------------i changed observed to True in both of these
            interaction=interaction.merge(mean_y_ai.rename('mean_a'),left_on=a,right_index=True)
            SSA = ((mean_y_ai-overall_mean)**2).sum() * nif_multiply_b  # nb * a sum of centered squares 
            mean_y_bj = three_col_XX_y.groupby(b,observed=True)[y].mean()
            interaction=interaction.merge(mean_y_bj.rename('mean_b'),left_on=b,right_index=True)
            SSB = ((mean_y_bj-overall_mean)**2).sum() * nij_multiply_a  # na * b sum of centered squares
        if unbalanced_interaction_sizes==True:
            mean_and_count_y_ai = three_col_XX_y.groupby(a, observed=True)[y].agg(['mean','size'])
            mean_and_count_y_ai.columns=['mean_a','size_a']
            SSA = ((mean_and_count_y_ai['mean_a'] - overall_mean)**2 * mean_and_count_y_ai['size_a']).sum()
            mean_and_count_y_bj = three_col_XX_y.groupby(b, observed=True)[y].agg(['mean','size'])
            mean_and_count_y_bj.columns=['mean_b','size_b']
            SSB = ((mean_and_count_y_bj['mean_b'] - overall_mean)**2 * mean_and_count_y_bj['size_b']).sum()

        # sum of squares for interaction between a & b    
        if not unbalanced_interaction_sizes:
            SSAB = ((interaction['mean'] - interaction['mean_a'] - interaction['mean_b'] + overall_mean)**2).sum() * nij
        if unbalanced_interaction_sizes==True:
            interaction=interaction.merge(mean_and_count_y_ai['mean_a'],left_on=a,right_index=True)
            interaction=interaction.merge(mean_and_count_y_bj['mean_b'],left_on=b,right_index=True)
            SSAB = ((interaction['mean'] - interaction['mean_a'] - interaction['mean_b'] + overall_mean)**2 * interaction['size']).sum()

        if verbose==True:  
            print(f"SST = SSA+SSB+SSAB+SSE: {SST == SSA+SSB+SSAB+SSE}\nnp.isclose(SST, SSA+SSB+SSAB+SSE): {np.isclose(SST, SSA+SSB+SSAB+SSE)}")
            print("SST:", SST)
            print("SSA + SSB + SSAB + SSE:", SSA + SSB + SSAB + SSE)
            print("Residual:", SST - (SSA + SSB + SSAB + SSE))
            print("If the residual error is large, try another model \ntwo_way_ANOVA_for_un_balanced_data(), which implements statsmodels.api\nor over/under sample the data.") 
        # degrees of freedom
        dof_a, dof_b, dof_ab, dof_e = a_number_of_groups-1, b_number_of_groups-1, (a_number_of_groups-1)*(b_number_of_groups-1), three_col_XX_y.shape[0]-(a_number_of_groups * b_number_of_groups)
        # mean squares
        MSA,MSB,MSAB,MSE =  SSA/dof_a, SSB/dof_b, SSAB/dof_ab, SSE/dof_e
        # F-statistics 
        FA,FB,FAB = MSA/MSE, MSB/MSE, MSAB/MSE
        # p values
        p_value_A, p_value_B, p_value_AB = scipy.stats.f.sf(FA,dof_a,dof_e), scipy.stats.f.sf(FB,dof_b,dof_e), scipy.stats.f.sf(FAB,dof_ab,dof_e)

        res= {a:p_value_A,b:p_value_B,'interaction':p_value_AB}
        return res
    
    # =========================================================================================================================================================================

    # A function that pairs combos based on inputs
    #it is used in kruskal_wallis and anova funtions that multiple many p-vlues based on the the combos
    def determine_column_combinations(self,data, numeric_columns:str|list|None=None,categoric_columns:str|list|None=None,categoric_target:str|list|None=None,numeric_target:str|list|None=None):
    
        """
        where categoric_columns and numeric_columns default to auto_detect if not [] or column(s) entered as str or list
        categoric_target and numeric_target can both be entered as str or list(s) of strings
        in case wehn neither catigoric_target or numeric_target is None:
            P-Value is eveluated when one OR the other is present in any combination, not one AND the the other
        """

        # DETERMINE THE CATEGORIC COLUMNS
        # base categoric w/o looking at target
        if categoric_columns == None : 
            categoric_columns=list(set(list(data.select_dtypes('object').columns)+list(data.select_dtypes('category').columns)))
        elif  isinstance(categoric_columns,str):
            categoric_columns=[categoric_columns]
        # look at categoric target
        if isinstance(categoric_target,str):
            categoric_target=[categoric_target]
        # ensure the categoric_target is in the categoric columns
        if categoric_target is not None:
            categoric_columns = list( set( categoric_columns + categoric_target ) )
        
        # DETERMINE THE NUMERIC COLUMNS
        # base numeric w/o looking at target
        if numeric_columns == None : 
            numeric_columns = list(data.select_dtypes('number').columns)
        elif isinstance(numeric_columns,str):
            numeric_columns=[numeric_columns]
        # look at numeric target
        if isinstance(numeric_target,str):
            numeric_target=[numeric_target]
        # ensure the numeric_target is in the numeric columns
        if numeric_target is not None:
            numeric_columns = list( set( numeric_columns + numeric_target ) )


        #conditional statements to consider target columns when defined or every valid combinanation otehrwise
        # no targets
        if (categoric_target is None) and (numeric_target is None):
            combinations=[(cat_col,num_col) for num_col in numeric_columns for cat_col in categoric_columns if cat_col!=num_col] 
        # numeric target only
        elif (categoric_target is None) and (numeric_target is not None):
            combinations=[(cat_col,num_col) for num_col in numeric_columns for cat_col in categoric_columns if ((cat_col!=num_col) and (num_col in numeric_target))]
        # categoric target only
        elif (categoric_target is not None) and (numeric_target is None):
            combinations=[(cat_col,num_col) for num_col in numeric_columns for cat_col in categoric_columns if ((cat_col!=num_col) and (cat_col in categoric_target))]
        # categoric and numeric targets
        elif (categoric_target is not None) and (numeric_target is not None):
            combinations=[(cat_col,num_col) for num_col in numeric_columns for cat_col in categoric_columns if ((cat_col!=num_col) and ((cat_col in categoric_target) or (num_col in numeric_target)))]
        # raise error if no condition is met
        else:
            raise ValueError("There seems to be an error in one or both of numeric_target and categoric_target parameters")
        #return list of combinations: [(cat,num),(cat2,num2), ..., (catn,numn)]
        return combinations

    # ========================================================================================================================================================================= 
    # one way tests: ANOVA and Kruskal Wallis
    # =========================================================================================================================================================================
    # ANOVA
    def one_way_ANOVA(self,two_col_df_x_y):
        """
        Where col in position [0] is catigorical and [1] is numeric
        returns np.nan when there arent enough observations or when there is no within-group variance
        """
        two_col_df_x_y=pd.DataFrame(two_col_df_x_y) 
        cols=two_col_df_x_y.columns
        x= cols[0]
        y= cols[1]
        number_of_groups=two_col_df_x_y[x].nunique()
        overall_mean=two_col_df_x_y[y].mean()

        #Sum of Squares Between
        SB_df=two_col_df_x_y.groupby(x,observed=True)[y].agg(['size','mean'])
        SSB=(((SB_df['mean']-overall_mean)**2)*SB_df['size']).sum()

        #Sum of Squares Within
        #untill broadcast level is supported @ https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.dataframe.subtract/#cudf.DataFrame.subtract
        #join is used instead of direct subtraction
        #map might be faster on smaller datasets than join
        SW_df=two_col_df_x_y.join(SB_df,on=x,validate='m:1')
        SW_df['observed-group_mean']=SW_df[y]-SW_df['mean']
        SW_df['observed-group_mean']=SW_df['observed-group_mean']**2
        SW_df=SW_df.groupby(x,observed=True)['observed-group_mean'].sum()
        #SW_df=SW_df**2
        SSW=SW_df.sum()
        del SW_df, SB_df

        #Degrees of Freedom
        dof_W=two_col_df_x_y.shape[0] - number_of_groups
        # return np.nan if there aren't enough observations for ANOVA
        if dof_W<=0: return np.nan
        dof_B=number_of_groups - 1

        #Mean Squares
        MSB=SSB/dof_B
        MSW=SSW/dof_W
        #return np.nan if MSW is 0: all values in groups are identical, ANOVA is meaningless
        if MSW == 0 or np.isclose(MSW, 0): return np.nan

        #F-statistic
        f_statistic = MSB/MSW 

        #P_value      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html#scipy.stats.f
        p_value = scipy.stats.f.sf(f_statistic,dof_B,dof_W)

        return p_value

    def test_all_cat_num_ANOVA(self,data, numeric_columns:str|list|None=None,categoric_columns:str|list|None=None,categoric_target:str|list|None=None,numeric_target:str|list|None=None):
        """
        where categoric_columns and numeric_columns default to auto_detect if not None or column(s) entered as str or list
        categoric_target and numeric_target can both be entered as str or list(s) of strings
        in case wehn neither catigoric_target or numeric_target is None:
            P-Value is eveluated when one OR the other is present in any combination, not one AND the the other
        """
        combinations = self.determine_column_combinations(data, numeric_columns=numeric_columns,categoric_columns=categoric_columns,categoric_target=categoric_target,numeric_target=numeric_target)
        if len(combinations)<1:
            return pd.DataFrame(columns=['category','numeric','P-value'])
        res_dict={}
        for combo in combinations:
            p=self.one_way_ANOVA(data[[*combo]])
            res_dict[(combo[0],combo[1])]=[p]
        res = pd.DataFrame(res_dict).T.reset_index(drop=False)
        res.columns=['category','numeric','P-value']
        self.one_way_df_ANOVA_overview=res
        return res

    # =========================================================================================================================================================================
    #ONE WAY KRUSKUL WALLIS TEST, similar to a onw way ANOVA but the assumptions make it more robust to non-normally distributed data
    #https://library.virginia.edu/data/articles/getting-started-with-the-kruskal-wallis-test
    #H = (12 / N(N + 1)) / sum(Ri² / ni) - 3(N + 1)
    # assumptions 1) all groups have the same distribution (else larger sample size), more groups require larger sample sizes, 
    # ========================================================================================================================================================================= 
    
    # Kruskal Wallis
    def one_way_kruskal_wallis(self,two_col_cat_num_df):
        """
        Where col in positions [0] is catigorical and [1] is numeric
        """
        
        data=pd.DataFrame(two_col_cat_num_df.copy())  #ensure CuDF if applicable. A check would be better, but this prototypes well
        data.columns=[f"temp_{(i+1)*10}" for i in range(len(data.columns))]
        cols=data.columns
        x = cols[0]
        y = cols[1]
        #assign ranks
        data['rank']=data[y].rank(method='average')
        #   where ni is each group size
        # and Ri is sum of ranks for group i
        grouped_data=data.groupby(x,observed=True)['rank'].agg(['sum','size'])
        #where k is number of groups and N is total observations
        k=grouped_data.shape[0]
        N=data.shape[0]
        #print(f"(12/(N*(N+1))): {(12/(N*(N+1)))}, ((grouped_data['sum']**2)/grouped_data['size']) {((grouped_data['sum']**2)/grouped_data['size'])}, (3*(N+1)) {(3*(N+1))}, np.sum(  ((grouped_data['sum']**2)/grouped_data['size']) - (3*(N+1))  ) {(  np.sum((grouped_data['sum']**2)/grouped_data['size']) - (3*(N+1))  )}")
        h_statistic =  (12/(N*(N+1)))   *  np.sum(  ((grouped_data['sum']**2)/grouped_data['size'])) - (3*(N+1))  
        dof = k-1
        p_value = scipy.stats.chi2.sf(h_statistic, dof)
        return p_value
    
    def test_all_cat_num_kruskal_wallis(self,data, numeric_columns:str|list|None=None,categoric_columns:str|list|None=None,categoric_target:str|list|None=None,numeric_target:str|list|None=None):
        """
        where categoric_columns and numeric_columns default to auto_detect if not None or column(s) entered as str or list
        categoric_target and numeric_target can both be entered as str or list(s) of strings
        in case wehn neither catigoric_target or numeric_target is None:
            P-Value is eveluated when one OR the other is present in any combination, not one AND the the other
        """
        combinations = self.determine_column_combinations(data, numeric_columns=numeric_columns,categoric_columns=categoric_columns,categoric_target=categoric_target,numeric_target=numeric_target)
        if len(combinations)<1:
            return pd.DataFrame(columns=['category','numeric','P-value'])
        res_dict={}
        for combo in combinations:
            p=self.one_way_kruskal_wallis(data[[*combo]])
            res_dict[(combo[0],combo[1])]=[p]
        res = pd.DataFrame(res_dict).T.reset_index(drop=False)
        res.columns=['category','numeric','P-value']
        self.one_way_kruskal_wallis_df_overview=res
        return res
    
    #=========================================================================================================================================================
    # a comparison function that supports either 'kruskal' or 'anova'
    #=========================================================================================================================================================
    
    def cat_num_column_comparison(self,data, alpha=0.05,keep_above_p:bool|None=False, numeric_columns:list|None=None,categoric_columns:list|None=None,numeric_target:str|list|None=None,categoric_target:str|list|None=None, test_method:str='kruskal'):
        """
        test_method can be of ('kruskal','anova')
        takes alpha as a parameter 
        if keep_above_p==False, observations with p_values<alpha are returned
        if keep_above_p==True p_values>=alpha are returned
        else all p_values are returned
        """
        if test_method=='kruskal':
            p_table=self.test_all_cat_num_kruskal_wallis(data,numeric_columns,categoric_columns,categoric_target,numeric_target)
        elif test_method=='anova':
            p_table=self.test_all_cat_num_ANOVA(data,numeric_columns,categoric_columns,categoric_target,numeric_target)            
        else:
            raise ValueError("Unknown test_method. test_method should be one of ('kruskal','anova')")
        if keep_above_p==False:
            return p_table.loc[p_table['P-value']<alpha].reset_index(drop=True)
        elif keep_above_p==True:
            return p_table.loc[p_table['P-value']>=alpha].reset_index(drop=True)
        else: 
            return p_table

    # ========================================================================================================================================================================= 
    # ========================================================================================================================================================================= 
    # ========================================================================================================================================================================= 
    # two sample t_tests
    def two_sample_t_tests(self,catx_numy_df):

        """
        
        Default is "welch's" t_test because it is robust to unequal varaince and unequal sample sizes

        """

        data=pd.DataFrame(catx_numy_df)
        cols=data.columns
        cat=cols[0]
        num=cols[1]
        df1=data.groupby(cat,as_index=False,observed=True)[num].agg(['mean','std','size'])
        nandf=df1.loc[(df1['size'] <= 1) & (df1['std'] <= 0)]
        nandf['subcat_1'],nandf['subcat_2'],nandf['P-value'],nandf['n_samples_1'],nandf['n_samples_2'] = nandf[cat], np.nan, np.nan, nandf['size'], np.nan
        nandf=nandf[['subcat_1','subcat_2','P-value','n_samples_1','n_samples_2']]
        df1 = df1.loc[(df1['size'] > 1) & (df1['std'] > 0)]
        df1=df1.reset_index(drop=True)
        combos = list(itertools.combinations(df1[cat].unique(), 2)) 
        merged=pd.DataFrame(combos,columns=['subcat_1','subcat_2'])
        merged=merged.merge(df1,how='left',right_on=cat,left_on='subcat_1')
        merged=merged.merge(df1,how='left',right_on=cat,left_on='subcat_2',suffixes=('1','2'))
        merged['std1']=merged['std1']**2
        merged['std2']=merged['std2']**2
        merged['t_score']= (merged['mean1']-merged['mean2']) /  np.sqrt( (merged['std1']/merged['size1'])+(merged['std2']/merged['size2']) )
        merged['dof']= ( (merged['std1']/merged['size1'])+(merged['std2']/merged['size2']) )**2 / ( ((merged['std1']/merged['size1'])**2 / (merged['size1']-1)) +((merged['std2']/merged['size2'])**2 / (merged['size2']-1)) )
        merged['P-value'] = 2 * scipy.stats.t.sf( np.abs(merged['t_score']),  merged['dof']  )
        merged=merged.rename(columns={'size1':'n_samples_1','size2':'n_samples_2'})
        merged=pd.concat([merged[['subcat_1','subcat_2','P-value','n_samples_1','n_samples_2']],nandf])
        merged['n_samples_2']=merged['n_samples_2'].astype(int)
        return merged

    def subcategory_similarities(self,catx_numy_df,alpha=0.05,return_similar=False,min_observations:int=None):
        data=self.two_sample_t_tests(catx_numy_df)
        data=data.dropna(subset=['subcat_2'])
        if return_similar==False:
            data=data.loc[data['P-value']<alpha]
        if min_observations is not None:
            data=data.loc[(data['n_samples_1']>min_observations)&(data['n_samples_2']>min_observations)]
        data=data.reset_index(drop=True)
        return data





