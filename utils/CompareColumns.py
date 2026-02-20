

import pandas as pd
import numpy as np

try:
    from .Utils_HypTests_and_Coefficients.ANOVA import ANOVA
    from .Utils_HypTests_and_Coefficients.Chi2 import Chi2
    from .Utils_HypTests_and_Coefficients.Coefficient import Coefficient
    from .Utils_HypTests_and_Coefficients.TTests import TTests
except:
    from Utils_HypTests_and_Coefficients.ANOVA import ANOVA
    from Utils_HypTests_and_Coefficients.Chi2 import Chi2
    from Utils_HypTests_and_Coefficients.Coefficient import Coefficient
    from Utils_HypTests_and_Coefficients.TTests import TTests

class CompareColumns(ANOVA, Chi2, Coefficient, TTests):

    def column_comparison(self,
                        df,
                        numnum_meth_alpha_above:tuple|list|None=('welch',0.05,False),
                        catnum_meth_alpha_above:tuple|list|None=('kruskal',0.05,False),
                        catcat_meth_alpha_above:tuple|list|None=('chi2',0.05,False),
                        numeric_columns:str|list|None=None,
                        categoric_columns:str|list|None=None,
                        numeric_target:str|list|None=None,
                        categoric_target:str|list|None=None ):
        """
        parameters:
        df: a pandas dataframe
        numnum_meth_alpha_above, catnum_meth_alpha_above, and catcat_meth_alpha_above take input of:
            None or a tuple with (test method, alpha threshold, and whether >= or < in relation to threshold or both)
            if tuple, values should be (string, float, boolean|None).
            Examples: ('chi2',0.05,False), ('anova',0.025,None), ('welch',0.01,True).
            where: 
                numnum_meth_alpha_above for a numeric-to-numeric comparison. Accepts methods of ('welch','student','pearson','spearman',kendall').
                catnum_meth_alpha_above for a categoric-to-numeric comparison. Accepts methods of ('kruskal','anova').
                catcat_meth_alpha_above for a categoric-to-categoric comparison. Accepts method of ('chi2'). 
        numeric_columns and categoric_columns accept manual column input. Otherwise columns are autodetected,
        numeric_target and categoric_target accept target columns. If either or both, only combinations involving targets will be considered
        """
        # a list to store datframes in. it will be used in a concat function 
        result_frames_to_concat = []

        #create boolean variables to use for managing targets
        is_cat_target, is_num_target = (categoric_target is not None), (numeric_target is not None)
        # if there are any targets at all
        if is_cat_target or is_num_target:  # there is cat or num targets or both
            if not is_num_target: # then there is a cat target and no num target
                include_numnum, include_catnum, include_catcat = False, True, True
            elif not is_cat_target:  # then there is a num target and no cat target
                include_numnum, include_catnum, include_catcat = True, True, False
            else:  # there are cat and num targets
                include_numnum, include_catnum, include_catcat = True, True, True
        else:  # there are no targets
            include_numnum, include_catnum, include_catcat = True, True, True

        # append relevant dataframes to the concat list
        if include_catnum and (catnum_meth_alpha_above is not None):
            #retrieve cat to num df
            if catnum_meth_alpha_above[0] not in ('anova','kruskal'):
                raise ValueError(f"Categoric to Numeric method not recognized. Expected one of ('anova','kruskal'). Recieved {catnum_meth_alpha_above[0]}",ValueError)
            catnum_df=self.cat_num_column_comparison(df,
                                                        alpha=catnum_meth_alpha_above[1],
                                                        keep_above_p=catnum_meth_alpha_above[2],
                                                        numeric_columns=numeric_columns,
                                                        categoric_columns=categoric_columns,
                                                        numeric_target=numeric_target,
                                                        categoric_target=categoric_target,
                                                        test_method=catnum_meth_alpha_above[0])  
            catnum_df=catnum_df.rename(columns={'category':'column_b','numeric':'column_a'})
            catnum_df['test']=catnum_meth_alpha_above[0]
            if catnum_df.shape[0]>0:
                result_frames_to_concat.append(catnum_df)
        if include_catcat and (catcat_meth_alpha_above is not None):
            # retrieve cat to cat df
            if catcat_meth_alpha_above[0] not in ('chi2'):
                raise ValueError(f"Categoric to Categoric method not recognized. Expected one of ('chi2'). Recieved {catcat_meth_alpha_above[0]}",ValueError)
            catcat_df=self.categorical_column_comparison(df,
                                                            alpha=catcat_meth_alpha_above[1],
                                                            keep_above_p=catcat_meth_alpha_above[2],
                                                            categoric_columns=categoric_columns,
                                                            categoric_target=categoric_target )
            catcat_df=catcat_df.rename(columns={'category_a':'column_a','category_b':'column_b'})
            catcat_df['test']=catcat_meth_alpha_above[0]
            if catcat_df.shape[0]>0:
                result_frames_to_concat.append(catcat_df)
        if include_numnum and (numnum_meth_alpha_above is not None):
            # retrieve num to num df
            if numnum_meth_alpha_above[0] in ('pearson','spearman','kendall'):
                numnum_df=self.num_num_column_coefficient_comparison(df,
                                                                    corr_threshold=numnum_meth_alpha_above[1],
                                                                    keep_above_corr=numnum_meth_alpha_above[2],
                                                                    numeric_columns=numeric_columns,
                                                                    target=numeric_target,
                                                                    corr_method=numnum_meth_alpha_above[0])
            elif numnum_meth_alpha_above[0] in ('welch','student'):
                numnum_df=self.num_num_column_t_test_comparison(df,
                                                                alpha=numnum_meth_alpha_above[1],
                                                                keep_above_p=numnum_meth_alpha_above[2],
                                                                numeric_columns=numeric_columns,
                                                                target=numeric_target,
                                                                t_test_method=numnum_meth_alpha_above[0])
                
            else:
                raise ValueError(f"Numeric to Numeric method not recognized. Expected one of ('pearson','spearman','kendall','welch','student'). Recieved {numnum_meth_alpha_above[0]}",ValueError)
            numnum_df=numnum_df.rename(columns={'numeric_1':'column_a','numeric_2':'column_b'})
            numnum_df['test']=numnum_meth_alpha_above[0]
            if numnum_df.shape[0]>0:
                result_frames_to_concat.append(numnum_df)
        possible_columns=['column_a','column_b','test','P-value','Correlation']
        if len(result_frames_to_concat)<1:
            return pd.DataFrame(columns=possible_columns)
        result=pd.concat(result_frames_to_concat)
        result=result[[col for col in possible_columns if col in result.columns]]
        return result