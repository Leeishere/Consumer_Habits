

import pandas as pd
import numpy as np

try:
    from .Utils_HypTests_and_Coefficients.Chi2 import Chi2
except:
    from Chi2 import Chi2

class UnivariateAnalysis(Chi2):

    def column_goodness_of_fit(self,
                        df,
                        cat_alpha_above:tuple|list=(0.05,False),
                        categoric_columns:str|list|None=None,
                        expected_probs:list|None=None):
        """
        parameters:
        df: a pandas dataframe
        cat_alpha_above:tuple|list=(0.05,False) 
            where index 0 is alpha, 
            and index 1 should be True to include H1, 
                False to include H0
                None for unfiltered
        categoric_columns:str|list|None=None is a list of columns to test. If None, columns will be infered
            hence, if None, it is nescesary that column datatypes are accurate
        expected_probs can be a list of probabilites to test against. None defaults to uniform distribution
            no further documentation on expected_probs available at this time
        """
        good_of_fit_df = self.test_all_cat_columns_chi_good_of_fit(df,columns=categoric_columns,expected_probs=expected_probs)
        if cat_alpha_above[1]==False:
            return good_of_fit_df.loc[good_of_fit_df['P-value']<cat_alpha_above[0]].reset_index(drop=True)
        if cat_alpha_above[1]==True:
            return good_of_fit_df.loc[good_of_fit_df['P-value']>=cat_alpha_above[0]].reset_index(drop=True)
        else:
            return good_of_fit_df
