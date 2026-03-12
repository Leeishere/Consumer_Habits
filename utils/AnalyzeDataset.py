

try:
    from .CompareColumns import CompareColumns
except:
    from CompareColumns import CompareColumns
try:
    from .Utils_HypTests_and_Coefficients.Chi2 import Chi2
except:
    from Utils_HypTests_and_Coefficients.Chi2 import Chi2
try:
    from .Utils_HypTests_and_Coefficients.Combinators import calculate_num_combinations
except:
    from Utils_HypTests_and_Coefficients.Combinators import calculate_num_combinations
try:
    from .PlotClass import PlotClass
except:
    from PlotClass import PlotClass

import pandas as pd
import numpy as np
from warnings import warn
from itertools import combinations

class AnalyzeDataset(CompareColumns, Chi2, PlotClass):

    def __init__(self,
                 numnum_meth_alpha_above_instructions:list|tuple=[('pearson',0.6,None),('spearman',0.6,None),('kendall',0.6,None)], # where t tests cannot share the parameter with correlation tests
                 catnum_meth_alpha_above_instructions:list|tuple=[('kruskal',0.05,None),('anova',0.05,None)], 
                 catcat_meth_alpha_above_instructions:list|tuple=[('chi2',0.05,None)],
                 good_of_fit_uniform_test_instrucions:list|tuple=(0.05,None),
                 concatinated_multivariate_header_divider:str='_&_',
                 continuous_ordinalized_suffix:str='-ADcont-Ordinalized',  #AD for AnalyzeDataset class, cont for continuous
                 continuous_binned_suffix:str='-ADcont-Binned',
                 categorical_ordinalized_suffix:str='-ADcat-Ordinalized',
                 auto_bin:bool=False, 
                 bin_instructions:dict=None,
                 multivariate_params:dict = {'numeric_targets':True,  
                                       'catigorci_targets':True,
                                       'max_n_combination_size':3,
                                       'max_n_combinations':50_000,
                                       'min_combo_size':2},
                 supercat_subcat_params:dict = {'max_evidence':0.2,  
                                           'test_all_both_ways':False}
                 ):                                       
        # FUNCTIONALITY NOT YET SUPORTED
        self.auto_bin                                   = auto_bin
        self.bin_instructions                           = bin_instructions
        self.continuous_ordinalized_suffix              = continuous_ordinalized_suffix
        self.continuous_binned_suffix                   = continuous_binned_suffix
        self.categorical_ordinalized_suffix             = categorical_ordinalized_suffix
        self.concatinated_multivariate_header_divider   = concatinated_multivariate_header_divider
        # IS SUPPORTED        
        # test instructions
        self.numnum_meth_alpha_above              = numnum_meth_alpha_above_instructions  # where t tests cannot share the parameter with correlation tests
        self.catnum_meth_alpha_above              = catnum_meth_alpha_above_instructions  # where variable is not like stat dataframe. dataframe has numric in column 0 and categoric in column 1
        self.catcat_meth_alpha_above              = catcat_meth_alpha_above_instructions
        self.good_of_fit_uniform_test_instrucions = good_of_fit_uniform_test_instrucions
        self.multivariate_params                  = multivariate_params
        self.supercat_subcat_params               = supercat_subcat_params

        # keep track targets that have been fit   ---> can be used to filter targets in tests for future suport of piecemeal fitting-- such as when revising assumption handling in tests
        self.has_called_fit_column_relationships                   = set() 
        self.has_called_fit_column_relationships_bool              = False 
        self.has_called_fit_multivariate_column_relationships      = set()
        self.has_called_fit_multivariate_column_relationships_bool = False
        self.has_called_fit_goodness_of_fit_uniform                = set()
        self.has_called_fit_goodness_of_fit_uniform_bool           = False
        self.has_called_fit_supercat_subcat_pairs                  = set()
        self.has_called_fit_supercat_subcat_pairs_bool             = False

        # UNIVARIATE
        #numeric univariate columns
        # PASS
        #categoric univariate columns
        self.reject_null_good_of_fit        = None
        self.fail_to_reject_null_good_of_fit= None
        # BIVARIATE  (used for plotting and for identifying relationships to each column as an individual target

        #numeric to numeric bivariate pairs
        self.above_threshold_corr_numnum    = None
        self.below_threshold_corr_numnum    = None
        #numeric to categoric bivariate pairs       WHERE (NUMERIC, CATEGORIC) IS ARRANGEMENT
        self.reject_null_catnum             = None
        self.fail_to_reject_null_catnum     = None
        #categoric to categoric bivariate pairs
        self.reject_null_catcat             = None
        self.fail_to_reject_null_catcat     = None
        self.supercategory_subcategory_pairs= []
        # MULTIVARIATE
        self.significant_multivariate_combinations = [] # where data form is: [ [[target,target_dtype],[list_combo_of_n_columns],test_type(s)], [], ...]
        #         .         .         .           .     # Its the same form derived in _prepare_target_plot_data using data stored in self.target_cols_with_significant_matches_and_not_significant for plotting

        # TARGET COLUMNS KEYS AND REJECT OR CORR ABOVE VALUE DICTIONARY
        # where each column is a key, and values are as follows
        # the term 'significant_relationship' is used to describe rejected null or correlation >= threshold
        # not_significant is used to describe instances when test(s) failed to reject the null, or correlation was below the threshold
        # THIS DICT CONTAINS MULTI-VARIATE
        # numeric to multi-categoric multivariate groups
        # categoric - categoric multiariate groups
        # for each column
        # a dict to store column by column info
        self.target_cols_with_significant_matches_and_not_significant = {}
    # a dict template to set all new columns with 
    def _blank_target_dict(self):
        """
        a template dict
        used to initiate columns in self.target_cols_with_significant_matches_and_not_significant = {}
        """
        return {
                'significant_numeric_relationships':[],
                'significant_numeric_tests':[],            # indexes align tests to 'significant_numeric_relationships' pairs
                'significant_categoric_relationships':[],  # can be single or combined columns
                'significant_categoric_tests':[],            # indexes align tests to 'significant_categoric_relationships' pairs
                'significant_categoric_combination_group_relationship':[],  #[[col1,col2,col3], [], ...]]
                'significant_categoric_combination_group_relationship_test_type':[],  #[test(s)_group_1, test(2)_group_2]
                'paired_to_a_supercategory':[],  # a list of columns it is partitioned by: its supercategory partitioner column
                'paired_to_a_subcategory':[],   # a list of columns that are partitioned by it: its subcategory groups column
                'target_dtype': [],  # can be one of 'numeric','categoric' to represent the data type of column_1 (in this case) ,
                'max_n_variates_paired_with':[0],   #  int(biggest lenght of combo checked, default: [0]
                'not_significant_numerics':[],
                'not_significant_categorics':[],  # where columns that are grouped together to form significant relationships are ( intended at some point to be )removed and added to 'significant_categoric_relationship' as group(s)
                'is_normal_or_uniform':[]   # for uniform tests: 'reject_uniform' or 'fail_to_reject_uniform'. For normal: UNSUPPORTED, but relevant in self._prepare_target_plot_data
                }
    #---------------------------------------------------------------------------------------------------------------------------------
    # HELPER FUNCTIONS 
    #---------------------------------------------------------------------------------------------------------------------------------
    #######################################################################################
    # A HELPER FUNCTION THAT RETURNS FAIL TO REJECT NULL, REJECT NULL, ABOVE CORRELATION THRESHOLD, AND/OR BELOW CORRELATION THRESHOLD
    # it processes one bivariate group at a time: numeric-to-numeric, numeric-to-categoric, or categoric-to-categoric
    #######################################################################################

    def _categorize_bivariate_tests_as_rej_or_failrej(self,
                                                     test_df:pd.DataFrame,
                                                     test_instructions:list|tuple):
        """
        where test_df is output from CompareColumns().multi_test_column_comparison(); 
            such as :
            test_df = CompareColumns().multi_test_column_comparison(
                                df,
                                numnum_meth_alpha_above=[('pearson',0.6,None),('spearman',0.6,None),('kendall',0.6,None)],
                                catnum_meth_alpha_above=[('kruskal',0.05,None),('anova',0.05,None)],
                                catcat_meth_alpha_above=[('chi2',0.05,None)])
        ------
        returns
        for rejected or correlation above threshold: 
            [[col_a,col_b,test(s)],[...],[...],...] 
            such as [['Purchase Amount (USD)', 'Purchase Amount (USD)_Binned', 'kendall-pearson-spearman'], ...]
        else: w/o test
        """
        # check vor valid test direction input. shold be list or list of nested lists/tuples
        if (not isinstance(test_instructions,list) ) and (not isinstance(test_instructions,tuple)):
            raise ValueError(f"expected a list or tuple such as ('test_type',float(threshold),bool|None). Recieved: {test_instructions}")
        if (not isinstance(test_instructions[0],list) ) and (not isinstance(test_instructions[0],tuple)):
            if isinstance(test_instructions[0],str) and isinstance(test_instructions[1],float) and (isinstance(test_instructions[2],bool) or test_instructions[2]==None):
                test_instructions=[test_instructions]

        # get rej null or above threshold coefficients
        list_of_tests = [i[0] for i in test_instructions]
        rej_or_corr_df = test_df.loc[test_df['test'].isin(list_of_tests)]
        failrej_or_below_corr_df = test_df.loc[test_df['test'].isin(list_of_tests)]
        for instructions in test_instructions:
            # determine which side of threshold is of interest in this tests 
            if (instructions[0] in ('pearson','spearman','kendall')) and ('Correlation' in test_df.columns):  # Correlation might not be included if targets are cat
                rej_or_corr_df = rej_or_corr_df.loc[~((rej_or_corr_df['test']==instructions[0])&(rej_or_corr_df['Correlation']<instructions[1]))]
                failrej_or_below_corr_df = failrej_or_below_corr_df.loc[~((failrej_or_below_corr_df['test']==instructions[0])&(failrej_or_below_corr_df['Correlation']>=instructions[1]))]
            elif ('P-value' in test_df.columns):   # P-value might not be included if all targets and columns are numeric
                rej_or_corr_df = rej_or_corr_df.loc[~((rej_or_corr_df['test']==instructions[0])&(rej_or_corr_df['P-value']>=instructions[1]))]
                failrej_or_below_corr_df = failrej_or_below_corr_df.loc[~((failrej_or_below_corr_df['test']==instructions[0])&(failrej_or_below_corr_df['P-value']<instructions[1]))]
        rej_or_corr_df['num_tests_per_pair'] = rej_or_corr_df.groupby(['column_a','column_b'])['test'].transform('count')
        mx=rej_or_corr_df['num_tests_per_pair'].max()
        if mx>1:
            def concat_test_types(num_tests,test_x,test_y):
                if num_tests>1:
                    return str(test_x)+'|'+str(test_y)
                else:
                    return str(test_x)

            rej_or_corr_df=rej_or_corr_df[['column_a','column_b','num_tests_per_pair','test']]
            new_rej_or_corr_df = rej_or_corr_df
            for match in range(2,mx+1): 
                new_rej_or_corr_df = new_rej_or_corr_df.merge(
                                        rej_or_corr_df,
                                        how='inner',
                                        on=['column_a','column_b','num_tests_per_pair'],
                                        suffixes=('_x', '_y')
                                        )
                new_rej_or_corr_df['test']=new_rej_or_corr_df.apply(lambda x: concat_test_types(x['num_tests_per_pair'],x['test_x'],x['test_y']),axis = 1)
                new_rej_or_corr_df=new_rej_or_corr_df[['column_a','column_b','num_tests_per_pair','test']]
            def split_tests(test):
                return list(set(test.split('|')))
            new_rej_or_corr_df['test'] = new_rej_or_corr_df['test'].apply(split_tests)
            new_rej_or_corr_df['len_test'] = new_rej_or_corr_df['test'].apply(lambda x: len(x))
            new_rej_or_corr_df = new_rej_or_corr_df.loc[new_rej_or_corr_df['num_tests_per_pair']==new_rej_or_corr_df['len_test']][['column_a','column_b','test']]
            def clean_test(test):
                return '-'.join(sorted(test))
            new_rej_or_corr_df['test']=new_rej_or_corr_df['test'].apply(clean_test)
            new_rej_or_corr_df=new_rej_or_corr_df.drop_duplicates(keep='first')
            rej_or_corr_df=new_rej_or_corr_df
            del new_rej_or_corr_df
        else:
            rej_or_corr_df = rej_or_corr_df[['column_a','column_b','test']].drop_duplicates(subset=['column_a','column_b'],keep='first')

        # get list such as [(column_a,column_b,test),(), ...] to return for rejected or corr above
        # and get temporary list such as [(column_a,column_b),(), ...] to filter non-rej and corr below
        rej_or_corr_cols_tests = rej_or_corr_df.to_numpy().tolist()
        filter_columns = rej_or_corr_df[['column_a','column_b']].to_numpy().tolist()

        # get fail to reject or below corr columns, but NOT TEST TYPE
        failrej_or_below_corr_cols_tests = failrej_or_below_corr_df[['column_a','column_b']].drop_duplicates(keep='first').to_numpy().tolist()
        failrej_or_below_corr_cols_tests = [i for i in failrej_or_below_corr_cols_tests if i not in filter_columns]
        return rej_or_corr_cols_tests, failrej_or_below_corr_cols_tests




    #######################################################################################
    # A FUNCTION TO PROCESS USE self._categorize_bivariate_tests_as_rej_or_failrej ITERATIVELY 
    # TO UPDATE         self.above_threshold_corr_numnum, self.below_threshold_corr_numnum, self.reject_null_catnum
    #       AND  self.fail_to_reject_null_catnum, self.reject_null_catcat, self.fail_to_reject_null_catcat 
    # AND TO CREATE A TARGET PROFILE FOR EACH VARIABLE: self.target_cols_with_significant_matches_and_not_significant
    #######################################################################################



        
    def _update_model_with_test_df_to_col_pairs_and_cols_as_targets(self,
                                                test_df:pd.DataFrame,
                                                targets:list  # a list of numeric and/or categoric columns. or []
                                                ):
        """
        a function that calls self._categorize_bivariate_tests_as_rej_or_failrej iteratively
        and updates class variables
        """

        test_instructions_list = [instruction for instruction in [self.numnum_meth_alpha_above,self.catnum_meth_alpha_above,self.catcat_meth_alpha_above] if instruction != None]
        
        for instructions in test_instructions_list:
            significant , not_significant = self._categorize_bivariate_tests_as_rej_or_failrej(test_df=test_df,
                                                                                            test_instructions=instructions)
                
            # use the test instrucitons to determine where to put each column: categorical or numeric
            samp_test=instructions[0][0]
    
            # store relationships and test for each column
            # map col_a and col_b to target col's value_dict in self.target_cols_with_significant_matches_and_not_significant
            
            # begin with significant relationships                                                                            
            if significant:
                if samp_test in ('pearson','spearman','kendall','welch','student'):
                    left_destination='significant_numeric_relationships'
                    left_test_destination = 'significant_numeric_tests'
                    right_destination='significant_numeric_relationships'
                    right_test_destination = 'significant_numeric_tests'
                    self.above_threshold_corr_numnum    = significant
                elif samp_test in ('kruskal','anova'):
                    left_destination='significant_numeric_relationships'
                    left_test_destination = 'significant_numeric_tests'
                    right_destination='significant_categoric_relationships'
                    right_test_destination = 'significant_categoric_tests'
                    self.reject_null_catnum             = significant
                elif samp_test in ('chi2'):
                    left_destination='significant_categoric_relationships'
                    left_test_destination = 'significant_categoric_tests'
                    right_destination='significant_categoric_relationships'
                    right_test_destination = 'significant_categoric_tests'
                    self.reject_null_catcat             = significant
                else:
                    raise ValueError(f'{samp_test} not recognized as a test type. Recognized are: pearson,spearman,kendall,welch,student,kendall,anova,chi2')
                for pair_test in significant:
                    # initialize targets in the dict if the aren't already initialized
                    left_col, right_col, test_dash_tests = pair_test[0], pair_test[1], pair_test[2]
                    if not targets or (left_col in targets) or (right_col in targets):
                        # left col
                        if not targets or (left_col in targets):
                            if left_col not in self.target_cols_with_significant_matches_and_not_significant.keys():
                                self.target_cols_with_significant_matches_and_not_significant[left_col]=self._blank_target_dict()
                                if left_destination=='significant_numeric_relationships':
                                    self.target_cols_with_significant_matches_and_not_significant[left_col]['target_dtype']=['numeric']
                                elif left_destination=='significant_categoric_relationships':
                                    self.target_cols_with_significant_matches_and_not_significant[left_col]['target_dtype']=['categoric']
                            self.target_cols_with_significant_matches_and_not_significant[left_col][right_destination].append(right_col)
                            self.target_cols_with_significant_matches_and_not_significant[left_col][right_test_destination].append(test_dash_tests)
                        # right col
                        if not targets or (right_col in targets):
                            if right_col not in self.target_cols_with_significant_matches_and_not_significant.keys():
                                self.target_cols_with_significant_matches_and_not_significant[right_col]=self._blank_target_dict()
                                if right_destination=='significant_numeric_relationships':
                                    self.target_cols_with_significant_matches_and_not_significant[right_col]['target_dtype']=['numeric']
                                elif right_destination=='significant_categoric_relationships':
                                    self.target_cols_with_significant_matches_and_not_significant[right_col]['target_dtype']=['categoric']
                            self.target_cols_with_significant_matches_and_not_significant[right_col][left_destination].append(left_col)
                            self.target_cols_with_significant_matches_and_not_significant[right_col][left_test_destination].append(test_dash_tests)
                    else:
                        continue
            
            # repeat for insignificant relationships
            if not_significant:  
                if samp_test in ('pearson','spearman','kendall','welch','student'):
                    left_destination='not_significant_numerics'
                    right_destination='not_significant_numerics'
                    self.below_threshold_corr_numnum    = not_significant
                elif samp_test in ('kruskal','anova'):
                    left_destination='not_significant_numerics'
                    right_destination='not_significant_categorics'
                    self.fail_to_reject_null_catnum     = not_significant
                elif samp_test in ('chi2'):
                    left_destination='not_significant_categorics'
                    right_destination='not_significant_categorics'  
                    self.fail_to_reject_null_catcat     = not_significant                     
                else:
                    raise ValueError(f'{samp_test} not recognized as a test type. Recognized are: pearson,spearman,kendall,welch,student,kendall,anova,chi2')      
                for pair_test in not_significant:
                    # initialize targets in the dict if the aren't already initialized
                    left_col, right_col = pair_test[0], pair_test[1]
                    if not targets or (left_col in targets) or (right_col in targets):
                        # left col
                        if not targets or (left_col in targets):
                            if left_col not in self.target_cols_with_significant_matches_and_not_significant.keys():
                                self.target_cols_with_significant_matches_and_not_significant[left_col]=self._blank_target_dict()
                                if left_destination=='not_significant_numerics':
                                    self.target_cols_with_significant_matches_and_not_significant[left_col]['target_dtype']=['numeric']
                                elif left_destination=='not_significant_categorics':
                                    self.target_cols_with_significant_matches_and_not_significant[left_col]['target_dtype']=['categoric']
                            self.target_cols_with_significant_matches_and_not_significant[left_col][right_destination].append(right_col)
                        # right col
                        if not targets or (right_col in targets):
                            if right_col not in self.target_cols_with_significant_matches_and_not_significant.keys():
                                self.target_cols_with_significant_matches_and_not_significant[right_col]=self._blank_target_dict()
                                if right_destination=='not_significant_numerics':
                                    self.target_cols_with_significant_matches_and_not_significant[right_col]['target_dtype']=['numeric']
                                elif right_destination=='not_significant_categorics':
                                    self.target_cols_with_significant_matches_and_not_significant[right_col]['target_dtype']=['categoric']
                            self.target_cols_with_significant_matches_and_not_significant[right_col][left_destination].append(left_col)
                    else:
                        continue

    #######################################################################################
    # A FUNCTION TO PROCESS CATEGORIC AND NUMERIC TARGETS
    #
    #######################################################################################
    def _combine_targets(self,numeric_target:list|tuple|str|None, categoric_target:list|tuple|str|None):
        """
        a helper function to process targets
        """            
        targets=[]
        if isinstance(numeric_target,str):
            numeric_target = [numeric_target]
        if numeric_target!=None:
            for targ in numeric_target:
                targets.append(targ)
        if isinstance(categoric_target,str):
            categoric_target = [categoric_target]
        if categoric_target!=None:
            for targ in categoric_target:
                targets.append(targ)
        return targets

    #######################################################################################
    # A FUNCTION TO CONCATINATE COLUMNS INTO ONE OBJECT TYPE COLUMN THAT CAN BE USED FOR TESTS
    #
    #######################################################################################


    def _concatenate_columns_axis_1(self,
                                    dataframe:pd.DataFrame, 
                                    column_combo:list|tuple,
                                    header_divider:str="_&_"):
        """
        takes a dataframe 
        list of columns in the dataframe that should be concatinated into one Series with dtype object
        a header divider as string that divides variable headers and variabel values
        returns pd.Series with name as concatinated headers
        """

        new_col_header = column_combo[0]
        result = dataframe[column_combo[0]]
        other_columns=column_combo[1:]
        for index in range(len(other_columns)):
            new_col_header = new_col_header + header_divider + other_columns[index]
            result = result.astype(str)+header_divider+dataframe[other_columns[index]].astype(str)
        result.name=new_col_header
        return result
    
    #######################################################################################
    # A FUNCTION THAT CONSIDERS COLUMN PAIRS ONE BY ONE TO DETECT SUPERCATEGORY-SUBCATEGORY RELATIONSHIPS
    # used in fit_organize_supercat_subcat_pairs() to fit/reorganize the model
    #######################################################################################

    def _are_supercat_subcats(self,
                                           data:pd.DataFrame,
                                           max_evidence:float=0.2,  
                                           pairs_list:list|tuple|None=None,
                                           test_all_both_ways:bool=False ):
                              
        """
        determines relationships
        where max_evidence is shannon_entropy, and less evidence is stronger support that there is a supercategory-subcategory relationship
        by default, this analyzed column pairs in self.fail_to_reject_null_catcat and updates the model, such as to remove super-subcat pairs
        and put them into self.supercat_subcat_pairs
        to override the default, input pairs_list, and the function will compute only input pairs
        test_all_both_ways: 
        if False: when one variable has 2* as many unique values as the other variable, the test is only one way
        if True they are tested in both directions

        returns a list of [[supercat, subcat], [], ...], where each pair is a supercat-subcat relationship
        and a list of [True, False, ...] that coresponds to input, or self.fail_to_reject_null_catcat
        """


        # determine what pairs to test
        if pairs_list is None:
            pairs = self.reject_null_catcat
        else: 
            pairs = pairs_list
        
        # stores [True, False, True, ...] for pairs in input pairs_list
        is_supsub = []
        # reorder list to indicate relationship: [[supercat, subcat], [], ...]
        # where new oder always has supercat first
        list_reordered = []
        for pair in pairs:
            if test_all_both_ways==False:
                cont=False
                if (data[pair[1]].nunique()/2) > data[pair[0]].nunique():
                    cont=True
                    supercat, subcat = pair[0], pair[1]
                    is_partitioned = (self.evidence_is_supercat_given_subcat(data, supercat, subcat)<=max_evidence)
                    is_supsub.append(is_partitioned)
                    if is_partitioned:
                        order=[supercat,subcat]
                        list_reordered.append(order)
                elif data[pair[1]].nunique() < (data[pair[0]].nunique()/2):
                    cont=True
                    supercat, subcat = pair[1], pair[0]                    
                    is_partitioned = (self.evidence_is_supercat_given_subcat(data, supercat, subcat)<=max_evidence)
                    is_supsub.append(is_partitioned)
                    if is_partitioned:
                        order=[supercat,subcat]
                        list_reordered.append(order)
                if cont==True:
                    continue
            first_way=self.evidence_is_supercat_given_subcat(data, pair[0], pair[1])
            first_way=(first_way<=max_evidence)
            second_way=self.evidence_is_supercat_given_subcat(data, pair[1], pair[0] )
            second_way=(second_way<=max_evidence)
            is_partitioned = ( first_way or second_way)
            is_supsub.append(is_partitioned)
            if is_partitioned:
                if first_way:
                    order=[pair[0], pair[1]]
                else:
                    order=[ pair[1], pair[0]]
                list_reordered.append(order)
        return list_reordered, is_supsub
    
    #---------------------------------------------------------------------------------------------------------------------------------
    # FIT FUNCTIONS: 
    # 1) fit_column_relationships
    # 2) fit_multivariate_column_relationships
    # 3) fit_supercat_subcat_pairs
    #---------------------------------------------------------------------------------------------------------------------------------    

    #######################################################################################
    # A FUNCTION TO FIT COLUMN RELATIONSHIPS BASED ON NULL HYPOTHESIS THRESHOLDS AND CORRELATIONS
    # numeric-to-numeric, numeric-to-categoric, or categoric-to-categoric, categoric_univaraite
    # NO multivariate beyond bivariate, but this must be run before the multivariate can be processed
    # target columns w/ significant and not-significant column relationships: self.target_cols_with_significant_matches_and_not_significant
    #######################################################################################

    def fit_column_relationships(self, 
                                 df: pd.DataFrame,
                                 numeric_columns=None,
                                 categoric_columns=None,
                                 numeric_target:list|tuple|str|None=None,
                                 categoric_target:list|tuple|str|None=None,
                                 ):
        """
        parameters:
            df: pd.DataFrame: the dataframe to process. It will be statistically analyzed
            numeric_columns, categoric_columns, numeric_target, categoric_target
                are None by default but can be specified with lists (list or string for targets)
        this function  updates: 
                self.reject_null_good_of_fit        
                self.fail_to_reject_null_good_of_fit
                self.above_threshold_corr_numnum   
                self.below_threshold_corr_numnum   
                self.reject_null_catnum             
                self.fail_to_reject_null_catnum     
                self.reject_null_catcat             
                self.fail_to_reject_null_catcat 
                    which can be all be used for plotting 
        it also updates   
                self.target_cols_with_significant_matches_and_not_significant
                    which provides per column relations that aid in ML
                        and can be used in to process multivariate relations beyond bivariate, 
                            but only bivariate is processid within this function
        """
        # compute the statistic df to get p-values and correlations
        test_df = self.multi_test_column_comparison(
                                df,
                                numnum_meth_alpha_above=self.numnum_meth_alpha_above,
                                catnum_meth_alpha_above=self.catnum_meth_alpha_above,
                                catcat_meth_alpha_above=self.catcat_meth_alpha_above,
                                numeric_columns=numeric_columns,
                                categoric_columns=categoric_columns,
                                numeric_target=numeric_target,
                                categoric_target=categoric_target 
                                )

        # identify target(s) if present
        targets = self._combine_targets(numeric_target=numeric_target, categoric_target=categoric_target)

        # compute significant and not-significant bivariate pairs -- w/ test for significant pairs
        self._update_model_with_test_df_to_col_pairs_and_cols_as_targets(test_df=test_df,
                                                                        targets=targets)
        # update class object that tracks variables that have been fit
        t_update = set(i for i in self.target_cols_with_significant_matches_and_not_significant.keys())
        self.has_called_fit_column_relationships.update(t_update)
        self.has_called_fit_column_relationships_bool=True

        return
    
    #######################################################################################
    # A FUNCTION TO TEST GOODNESS OF FIT FOR UNIFORM DISTRIBUTION
    # UPDATES   self.reject_null_good_of_fit AND self.fail_to_reject_null_good_of_fit
    #######################################################################################
        
    def fit_goodness_of_fit_uniform(self,
                                    df:pd.DataFrame,
                                    categoric_columns:str|list|None=None):
        
        # test categorical univariate agianst a uniform distribution
        good_of_fit_uniform_df = self.filterable_all_column_goodness_of_fit(
                                                                            df,
                                                                            cat_alpha_above=self.good_of_fit_uniform_test_instrucions,
                                                                            categoric_columns=categoric_columns,
                                                                            expected_probs=None)
        good_of_fit_threshold=self.good_of_fit_uniform_test_instrucions[0]
        self.reject_null_good_of_fit        = list(good_of_fit_uniform_df.loc[good_of_fit_uniform_df['P-value']<good_of_fit_threshold]['category'].values)
        self.fail_to_reject_null_good_of_fit= list(good_of_fit_uniform_df.loc[good_of_fit_uniform_df['P-value']>=good_of_fit_threshold]['category'].values)
        
        # keep track of columns that have been fit
        fail_set, rej_set = set(i for i in self.fail_to_reject_null_good_of_fit), set(i for i in self.reject_null_good_of_fit)
        self.has_called_fit_goodness_of_fit_uniform.update(fail_set)
        self.has_called_fit_goodness_of_fit_uniform.update(rej_set)
        self.has_called_fit_goodness_of_fit_uniform_bool=True

        # update target dict: self.target_cols_with_significant_matches_and_not_significant
        for col in self.reject_null_good_of_fit:
            if self.target_cols_with_significant_matches_and_not_significant.get(col,None)==None:
                self.target_cols_with_significant_matches_and_not_significant[col]=self._blank_target_dict()
            self.target_cols_with_significant_matches_and_not_significant[col]['is_normal_or_uniform'].append('reject_uniform')
        for col in self.fail_to_reject_null_good_of_fit:
            if self.target_cols_with_significant_matches_and_not_significant.get(col,None)==None:
                self.target_cols_with_significant_matches_and_not_significant[col]=self._blank_target_dict()
            self.target_cols_with_significant_matches_and_not_significant[col]['is_normal_or_uniform'].append('fail_to_reject_uniform')
        return  # exit the function

    #######################################################################################
    # A FUNCTION TO FIT MULTIVARIATE COLUMN RELATIONSHIPS FOR COLUMNS THAT WERE NOT INCLUDED IN BIVARIATE RELATIONSHIPS IN fit_column_relationships()
    # This concatinates categoricat variables and performs Chi2, ANOVA, and Kruskal-Wallis tests
    #######################################################################################

    def fit_multivariate_column_relationships(self,
                                              df:pd.DataFrame,
                                              targets:list|tuple|str|None,
                                              numeric_targets:bool=True,  # if True, targets are from self.target_cols_with_significant_matches_and_not_significant
                                              catigorci_targets:bool=True,  # if True, targets are from self.target_cols_with_significant_matches_and_not_significant
                                              max_n_combination_size:int|None=3,
                                              max_n_combinations:int|None=50_000,
                                              min_combo_size:int=2
                                              ):
        """
        this iterates through 
            self.target_cols_with_significant_matches_and_not_significant = { 
                    column_1:{'not_significant_categorics':[]  
                             'target_dtype': []  # can be one of ['numeric'],['categoric'] to represent the data type of column_1 (in this case)
                             },
        and removes each columns from 'not_significant_categorics' when:
            the colums are in categorical combination(s) that rejects the null hypothosis, [columns are removed from consideration after all combos in combo_size have been considered, hence, it may be that many are in several combos of the same size]
                and creates keys and puts those columns into lists that get nested in: 
                self.target_cols_with_significant_matches_and_not_significant = {
                        column_1:{'significant_categoric_combination_group_relationship':[[combo,group,1],[col1,col2,col3]]  
                                'significant_categoric_combination_group_relationship_test_type':[test(s)_group_1, test(2)_group_2]  # the tests are stored in the same index location as the group combo
                                },
        parameters:
                    targets:list|tuple|str|None --> target columns. Each target will be tested/compared in relation to combinations: 
                            Combinations are a range of [min_combo_size, max_n_combination_size] concatinated categorical variable
                            NOTE: if targets is not None, targets override parameters: numeric_targets and categoric_targets
                            if not None, only targets will be considered
                    The following are not used when targets is not None: 
                            numeric_targets:bool=True, --> give autodetect instructions. If False, numeric targets won't be included
                            categoric_targets:bool=True, -->  give autodetect instructions. If False, categoric targets won't be included
                            by default both are True, hence all columns are considered as targets
                    The following limit the number of combinations and combination sizes
                            max_n_combination_size:int|None, --> Set the max lenght of individual combinations
                            max_n_combinations:int|None=50_000 --> The max number of combinations, per variable for each combination size in the range [2, max_n_combination_size]
                                    the range is computed from smalled to largest combination size. 
                                    If the combination size is large enough to excede max_n_combinations, iteration stops and ONLY lower combination sizes are returned
                    min_combo_size defaults to minimum lenght of combos == 2, but it can take values other than 2
                    mulitvariate_test_significant_in_many_groups False removes any column from potential columns as soon as it is tested in a group that has a significant relationship 
        """
        # fit_column_relationships() has to have been called first:
        if self.has_called_fit_column_relationships_bool==False:
            raise ValueError("fit_column_relationships() needs to run before this function.")
        # if targets is an empty list convert it to None
        if not targets:
            targets=None
        # make a list of target variables if is/are string(s)
        if isinstance(targets,str):
            targets=[targets]
        if targets==None:
            if (numeric_targets==True) and (catigorci_targets==True):
                targets = list(self.target_cols_with_significant_matches_and_not_significant.keys())
            elif (numeric_targets==True) and (catigorci_targets==False):
                targets = [k for k,v in self.target_cols_with_significant_matches_and_not_significant.items() if self.target_cols_with_significant_matches_and_not_significant[k]['target_dtype']==['numeric']]
            elif (numeric_targets==False) and (catigorci_targets==True):
                targets = [k for k,v in self.target_cols_with_significant_matches_and_not_significant.items() if self.target_cols_with_significant_matches_and_not_significant[k]['target_dtype']==['categoric']]
            else:
                raise ValueError('One of input parameters numeric_targets or catigorci_targets is not a boolean value')
            
        # update class object that tracks targets that have been fit  
        if isinstance(targets,str):targets=[targets]
        t_update=set( i for i in targets)  
        self.has_called_fit_multivariate_column_relationships.update(t_update)
        self.has_called_fit_multivariate_column_relationships_bool=True
        #########################################################################################
        #########################################################################################
        ## CONSIDER A VECTORIZED VERSION AS AN ALTERNATIVE PARAMETER. 
        ## AS IT IS, IT ITERATES THROUGH COMBOS.
        ## IT COULD TAKE CHUNKS W CHUNCKSIZE PARAMETER
        ## SUCH AS BIG CHUNKS IN CASE OF CLOUD COMPUTE
        #########################################################################################
        ######################################################################################### 
        # iterate through targets list
        for target in targets:    
            # compute pairs, then remove those columns and compute 3's with left over, etcetera for 4's, 5's sorforth if/when parameters consider byond 2's
            #iterate while max num possible combos is not exceded: max_n_combinations
            biggest_combo_size_checked=0
            population_size = len(self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'])
            curr_combo_size = min_combo_size
            curr_n_possible_combinations = calculate_num_combinations(population_size,curr_combo_size) +1 # plus one because the try/except block will handle it
            while (curr_combo_size<=max_n_combination_size) and (curr_n_possible_combinations<=max_n_combinations) and (population_size>=curr_combo_size):
                # BEGIN INNER LOOP TO ITERATE THROUGH THESE COMBOS
                # iterate through categoric variables that have not yet been included in a reject null test in previous combo sizes
                combo_generator = combinations(list(set(self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'])),curr_combo_size)
                seen_and_significant={}
                for nth_combo in range(curr_n_possible_combinations):
                    try:
                        curr_combo = next(combo_generator)
                    except:
                        break
                    curr_combo = list(curr_combo)

                    # concatinate the curr_combo columns into one series, pair it in a dataframe with the target,c
                    header_divider                                                 = self.concatinated_multivariate_header_divider  # how concatinated multivariate tests' headers will seperate columns
                    df_to_use_in_concat_func                                       = df[curr_combo].copy()
                    concated_feature_col                                           = self._concatenate_columns_axis_1(dataframe=df_to_use_in_concat_func,
                                                                                                                    column_combo=curr_combo,
                                                                                                                    header_divider=header_divider)
                    target_and_concated_second_col_df                              = df[[target]].copy()
                    concated_feature_col_header                                    = concated_feature_col.name
                    target_and_concated_second_col_df[concated_feature_col_header] = concated_feature_col.astype('object')
                    # Hypothosis test target_and_concated_second_col_df 
                    if self.target_cols_with_significant_matches_and_not_significant[target]['target_dtype'] == ['numeric']:
                        test_instructions = self.catnum_meth_alpha_above
                        test_df = self.multi_test_column_comparison(
                                        target_and_concated_second_col_df,
                                        numnum_meth_alpha_above=None,
                                        catnum_meth_alpha_above=self.catnum_meth_alpha_above,
                                        catcat_meth_alpha_above=None,
                                        numeric_columns=None,
                                        categoric_columns=None,
                                        numeric_target=target,
                                        categoric_target=None 
                                        )
                    elif self.target_cols_with_significant_matches_and_not_significant[target]['target_dtype'] == ['categoric']: 
                        test_instructions = self.catcat_meth_alpha_above
                        test_df = self.multi_test_column_comparison(
                                        target_and_concated_second_col_df,
                                        numnum_meth_alpha_above=None,
                                        catnum_meth_alpha_above=None,
                                        catcat_meth_alpha_above=self.catcat_meth_alpha_above,
                                        numeric_columns=None,
                                        categoric_columns=None,
                                        numeric_target=None,
                                        categoric_target=target 
                                        )
                    else:
                        raise ValueError(f"No data type detected for {target}: self.target_cols_with_significant_matches_and_not_significant[target]['target_dtype'] should be one of ['categoric'] or ['numeric']")
                    if test_df.shape[0]>0: 
                        significant , not_significant = self._categorize_bivariate_tests_as_rej_or_failrej(test_df=test_df,
                                                                        test_instructions=test_instructions)
                        if significant:  #significant looks like: [[col_a,col_b,test(s)],[...],[...],...] But in this case shorter
                            # the columns are already known: curr_combo: so just extract the test
                            curr_test=significant[0][2]
                            self.target_cols_with_significant_matches_and_not_significant[target]['significant_categoric_combination_group_relationship'].append(curr_combo)
                            self.target_cols_with_significant_matches_and_not_significant[target]['significant_categoric_combination_group_relationship_test_type'].append(curr_test)
                            # same as in self._prepare_target_plot_data(): [[target_column,'categoric'],col,test]
                            item_to_append_to_overall_model_reject_combo_matches = [[target,self.target_cols_with_significant_matches_and_not_significant[target]['target_dtype'][0]],curr_combo,curr_test]  
                            self.significant_multivariate_combinations.append(item_to_append_to_overall_model_reject_combo_matches)
                            # store columns that won't be passed to the next combo size
                            if seen_and_significant:
                                curr_combo = set(curr_combo)
                                seen_and_significant.update(curr_combo)
                            else:
                                seen_and_significant=set(curr_combo)
                # after each size, remove combos that have already been in a significant group from 'not_significant_categorics' 
                # the idea is that, otherwise, these groups would re-apear with additional collumns that may not really contribute 
                try: 
                    for var in list(seen_and_significant):
                        var_index=self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'].index(var)
                        discard = self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'].pop(var_index)
                except:
                    for var in list(seen_and_significant):
                        var_index=self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'].index(var)
                        self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'] = self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'][:var_index] + self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'][var_index+1:]
                

                #update biggest_combo_size_checked, curr_combo_size, curr_n_possible_combinations, and population_size
                biggest_combo_size_checked=curr_combo_size
                curr_combo_size+=1
                curr_n_possible_combinations=calculate_num_combinations(population_size,curr_combo_size) +1
                population_size=len(self.target_cols_with_significant_matches_and_not_significant[target]['not_significant_categorics'])
            # if the loop was never entered 
            if curr_combo_size==min_combo_size:
                if curr_n_possible_combinations > max_n_combinations:
                    # consider raising max_n_combinations
                    raise ValueError(f"No combinations considered.\nConsider increasing max_n_combinations.\n max_n_combinations == {max_n_combinations}, but found {curr_n_possible_combinations} combinations @ min combo size == {min_combo_size}.")
                elif population_size<curr_combo_size:
                    warn(f"No combinations considered. population_size == {population_size}, but minimum combo size is {min_combo_size}")
                # indicate that this target never entered the loop, and biggest size of combos that were checked: 0
                self.target_cols_with_significant_matches_and_not_significant[target]['max_n_variates_paired_with']=[0]
            else:
                #add a key,value pair to indicate that this target did enter the loop, and biggest size of combos tested
                self.target_cols_with_significant_matches_and_not_significant[target]['max_n_variates_paired_with']=[biggest_combo_size_checked]

        return        
    
    #######################################################################################
    # A FUNCTION TO REMOVE SUBCAT PAIRS FROM REGECT NULL CAT-TO-CAT RESULTS
    # removes from rej null catcat pairs and adds to self.supercategory_subcategory_pairs
    # removes rej null categoric from individual categorical targets and adds to paired_to_supercat (if target is subcat), or to paired_to_subcat(if target is supercat)
    #######################################################################################
        
    def fit_supercat_subcat_pairs(self,
                                    data:pd.DataFrame,
                                    max_evidence:float=0.2,  
                                    test_all_both_ways:bool=False,
                                     pairs_list:list|tuple|None=None ):
        """
        calls:  _are_supercat_subcats()
        to detect spercategory-subcategory relationships
        then updates class objects:
            removes from rej null catcat pairs and adds to self.supercategory_subcategory_pairs
            removes rej null categoric from individual categorical targets and adds to paired_to_supercat (if target is subcat), or to paired_to_subcat(if target is supercat)
        where test_all_both_ways is passed to _are_supercat_subcats() 
            if False, column pairs with unique variable rations 1/2 or smaller are only tested in one directions        
        """

        # fit_column_relationships() has to have been called first:
        if self.has_called_fit_column_relationships_bool==False:
            raise ValueError("fit_column_relationships() needs to run before this function.")

        sup_subs, true_false_list =  self._are_supercat_subcats(data,
                                           max_evidence=max_evidence,  
                                           pairs_list=pairs_list,
                                           test_all_both_ways=test_all_both_ways ) 

        new_rej_null=[] 
        # reset   self.reject_null_catcat to exclude supercat-subcat pairs
        for tf_bool, cols in zip(true_false_list,self.reject_null_catcat):
            if tf_bool == False:
                new_rej_null.append(cols)

            # store record of fit columns in class
            set_cols = set(i for i in cols)
            self.has_called_fit_supercat_subcat_pairs.update(set_cols)
            self.has_called_fit_supercat_subcat_pairs_bool=True
        # THIS REMOVES SUPER-SUBCAT PAIRS FROM SELF.REJECT_NULL_CATCAT  the next loop places them in a super-subcat object: self.supercategory_subcategory_pairs
        self.reject_null_catcat  =  new_rej_null
        # reset per target
        for cols in sup_subs:
            # append pairs to class list
            self.supercategory_subcategory_pairs.append(cols)

            #identify each
            super, sub = cols[0], cols[1]
            
            # reset supercat  [remove from reg null and add to paired to a subcategory]
            self.target_cols_with_significant_matches_and_not_significant[super]['paired_to_a_subcategory'].append(sub)
            sub_index=self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_relationships'].index(sub)
            try:
                discard_col=self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_relationships'].pop(sub_index)
                discard_test=self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_tests'].pop(sub_index)
            except:
                if sub_index<(len(self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_relationships'])-1):
                    self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_relationships']=self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_relationships'][:sub_index]+self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_relationships'][sub_index+1:]
                    self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_tests']        =        self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_tests'][:sub_index]+self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_tests'][sub_index+1:]
                else:
                    self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_relationships']=self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_relationships'][:sub_index]
                    self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_tests']        =        self.target_cols_with_significant_matches_and_not_significant[super]['significant_categoric_tests'][:sub_index]
            
            # reset subcat [remove from rej null and add to paired to a supercategory]
            self.target_cols_with_significant_matches_and_not_significant[sub]['paired_to_a_supercategory'].append(super)
            super_index=self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_relationships'].index(super)
            try:
                discard_col=self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_relationships'].pop(super_index)
                discard_test=self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_tests'].pop(super_index)
            except:
                if super_index<(len(self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_relationships'])-1):
                    self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_relationships']=self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_relationships'][:super_index]+self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_relationships'][super_index+1:]
                    self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_tests']        =        self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_tests'][:super_index]+self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_tests'][super_index+1:]
                else:
                    self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_relationships']=self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_relationships'][:super_index]
                    self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_tests']        =        self.target_cols_with_significant_matches_and_not_significant[sub]['significant_categoric_tests'][:super_index]
        return
    

    """
    #######################################################################################
    # A FUNCTION THAT WRAPS FIT FUNCTIONS
    # FUNC 1) fit_column_relationships(self, 
                                 df: pd.DataFrame,
                                 numeric_columns=None,
                                 categoric_columns=None,
                                 numeric_target=None,
                                 categoric_target=None,
                                 )
    # FUNC 2) fit_multivariate_column_relationships(self,
                                              df:pd.DataFrame,
                                              targets:list|tuple|str|None,
                                              numeric_targets:bool=True,
                                              catigorci_targets:bool=True,
                                              max_n_combination_size:int|None=3,
                                              max_n_combinations:int|None=50_000,
                                              min_combo_size:int=2
                                              )
    # FUNC 3) fit_supercat_subcat_pairs(self,
                                           data:pd.DataFrame,
                                           max_evidence:float=0.2,  
                                           pairs_list:list|tuple|None=None,
                                           test_all_both_ways:bool=False )
    # #######################################################################################
    """

    def fit_full_dataset_analysis(self,
                              data:pd.DataFrame,
                                numeric_columns=None,
                                categoric_columns=None,
                                numeric_target=None,
                                categoric_target=None,
                                fit_good_of_fit:bool=True,
                                fit_multivariates:bool=False,
                                fit_supercat_subcats:bool=False
                                ): 
        """
        parameters for manual entry of columns, and for targets
            numeric_columns=None,
            categoric_columns=None,
        """


        self.fit_column_relationships( df=data,
                                numeric_columns=numeric_columns,
                                categoric_columns=categoric_columns,
                                numeric_target=numeric_target,
                                categoric_target=categoric_target
                                )
        if fit_good_of_fit==True:
            self.fit_goodness_of_fit_uniform(
                                        df=data,
                                        categoric_columns=categoric_columns)        
        if fit_multivariates==True:
            # identify target(s) if present
            targets = self._combine_targets(numeric_target=numeric_target, categoric_target=categoric_target)
            fit_multivariate_args=self.multivariate_params
            self.fit_multivariate_column_relationships(df=data,
                                                targets=targets,
                                                **fit_multivariate_args
                                                )
        if fit_supercat_subcats==True:
            fit_super_subcat_args=self.supercat_subcat_params
            self.fit_supercat_subcat_pairs(data=data,
                                            **fit_super_subcat_args) 
        



    #-------------------------------------------------------------------------------------------------------------------------------
    # HELPER FUNCTIONS AND A FINAL FUNCTION: column_relationships_df(),  TO MANIPULATE RETURN A DATAFRAME FROM self.target_cols_with_significant_matches_and_not_significant
    # the dataframe only includes data on significan comparisons
    #-------------------------------------------------------------------------------------------------------------------------------    

    def _concate_significant_lists(self,
                                input_dict):
        """
        concatenates so that there only 2 columns: columns and tests
                all of: 
                        'significant_numeric_relationships':[],
                        'significant_numeric_tests':[], 
                        'significant_categoric_relationships':[], 
                        'significant_categoric_tests':[], 
                        'significant_categoric_combination_group_relationship':[], 
                        'significant_categoric_combination_group_relationship_test_type':[],
                are put into two lists:
                        'SignificantCounterParts' and 'Tests'
        """

        concated_cols, concated_tests = [], []
        cols_tests = [('significant_numeric_relationships','significant_numeric_tests'),('significant_categoric_relationships','significant_categoric_tests'),('significant_categoric_combination_group_relationship','significant_categoric_combination_group_relationship_test_type')]
        new_dict = input_dict.copy()
        for keys in cols_tests:
            concated_cols+=input_dict[keys[0]]
            try:
                del new_dict[keys[0]]
            except:
                discard = new_dict.pop(keys[0],None)
            concated_tests+=input_dict[keys[1]]
            try:
                del new_dict[keys[1]]
            except:
                discard = new_dict.pop(keys[1],None)
        new_dict['SignificantCounterPart(s)']=concated_cols
        new_dict['Tests']=concated_tests
        return new_dict

        

    def _max_vec_in_dict(self,
                        input_dict):
        """
        where all values in the dict should be [] or ()
        return the lenght of the longest value
        """
        mx=-np.inf
        for v in input_dict.values():
            if len(v)>mx:
                mx=len(v)
        return mx

    def _null_pad_vector(self,
                        final_length:int, 
                        curr_vector:list, 
                        pandas_na:bool=True):
        """
        pad vectors with pd.NA if pandas_na == True
        else np.nan
        returns the curr_vector, right/bottom padded
        """
        num_nans_needed = final_length - len(curr_vector)
        if not isinstance(curr_vector,list):
            curr_vector = list(curr_vector)
        if pandas_na==True:
            pad = [pd.NA]*num_nans_needed
        else:
            pad = [np.nan]*num_nans_needed
        return curr_vector + pad

    def _value_pad_vector(self,
                        final_length:int, 
                        curr_vector:list):
        """
        returns the curr_vector, such as [2], padded such as [2,2,2], where the value repeats
        """
        num_nans_needed = final_length - len(curr_vector)
        if not isinstance(curr_vector,list):
            curr_vector = list(curr_vector)
        pad = [curr_vector[0]]*num_nans_needed
        return curr_vector + pad
        
    def _prep_key_value_for_df(self,
                            key,
                            value ):
        """
        takes a key, value pair from AD.target_cols_with_significant_matches_and_not_significant
        and returns a dict that:
            includes the key
            and has all values in listlike arrays of equal lenght
        """
        # concatinate significant counter columns and concatinate the tests
        val = self._concate_significant_lists(value)  # this uses a copy() internaly and returns a new dict
        # put the target column into the dict
        val['Target']=[key]
        # remove unwanted columns
        try:
            del val['not_significant_numerics'], val['not_significant_categorics']
        except:
            val.pop('not_significant_numerics',None)
            val.pop('not_significant_categorics',None)
        # retrieve the max vector lenght
        mx_len = self._max_vec_in_dict(val)  #max(1,...) is unescesary, because val['Target'] is in the dict
        for k,v in val.items():        
            # paw w original value
            if k in ('Target','target_dtype','max_n_variates_paired_with'):
                val[k]=self._value_pad_vector(mx_len, v)
            # NaN pad
            else:
                val[k]=self._null_pad_vector(mx_len, v, pandas_na=True)
        return val

    def _dict_to_df(self,
                    target_dict, 
                    col_name_map:dict={
                                            'SignificantCounterPart(s)':'SignificantCounterPart(s)',
                                            'Tests':'Test(s)',
                                            'Target':'Target',
                                            'paired_to_a_supercategory':'PartitionedBy',
                                            'paired_to_a_subcategory':'Partitions',
                                            'target_dtype': 'TargetDType', 
                                            'max_n_variates_paired_with':'MaxLenComboComparedTo'
                                            }
                                            ):
        """
        takes output from _prep_key_value_for_df(): parameter: target_dict [ the dict for that target varable]
        renames columns to fit headers
        and returns a dataframe with index['Target','TargetDType','MaxLenComboComparedTo']
        and values [SignificantCounterPart(s), Test(s), PartitionedBy, Partitions]
        """
        # create df
        target_df = pd.DataFrame.from_dict(target_dict)
        # rename columns
        target_df = target_df.rename(columns=col_name_map,inplace=False)
        # set index 
        target_df = target_df.set_index(['Target','TargetDType','MaxLenComboComparedTo'])
        # return with column in human friendly order
        return target_df[['SignificantCounterPart(s)', 'Test(s)', 'PartitionedBy', 'Partitions']]

    def column_relationships_df(self):
        """
        processes self.target_cols_with_significant_matches_and_not_significant
        and returns a dataframe that includes all target columns, datatypes, max_multivairaes paired with, 
                                                    all significant relationships, tests used, 
                                                    supercat_cols_that partition, 
                                                    and cols its subcats partition
        calls self._prep_key_value_for_df() and self._dict_to_df() on k, v, then concatinates results
        returns a dataframe with 
            multiindex:
                'Target': the target column
                'TargetDType': data type of the target column
                'MaxLenComboComparedTo': Indicates the max concatinated-combo size compared to. 
                                         This can vary based on the number of combinations and parameters that manage compute.
            columns:
                'SignificantCounterPart(s)': single strings that are column headers, or lists of headers
                'Test(s)': indicates the tests used to determine relationship status 
                'PartitionedBy': colum(s) with values that partition the target variable 
                'Partitions': column(s) the target variable's values partition                    
        """
        
        concat_list = []
        for k,v in self.target_cols_with_significant_matches_and_not_significant.items():
            curr_dict = self._prep_key_value_for_df(k,v)
            curr_df   = self._dict_to_df(curr_dict)
            concat_list.append(curr_df)
        print("Where 'MaxLenComboComparedTo' can vary depending on compute limits and number of possible combinations in combo sizes.\n'PartitionedBy' and 'Partitions' indicate supercategory-subcategory relationships that partition or are partitioned.\n'SignificantCounterPart(s)' is a combination or single column that shares a significant relationship according to the test(s).")
        return pd.concat(concat_list)


    ##############################################################################################################################
    ##############################################################################################################################
    # VISUALIZING THE OUPUT
    ##############################################################################################################################
    ##############################################################################################################################


    #######################################################################################
    # A FUNCTION TO PLOT UNIVARIATE WHERE REJECT NULL GOODNESS OF FIT FOR UNIFORM DISTRIBUTION
    # PLOTS VALUES STORED IN self.reject_null_good_of_fit
    #######################################################################################

    def plot_non_uniform_categorical(self,
                                        data:pd.DataFrame,
                                        categorical:list|tuple|None=None,
                                        proportions:bool=False,
                                        n_wide:int|tuple|list=(6,40,4),
                                        super_title:str|None="Univariate Categorical Variables - Reject Good-Of-Fit for Uniform"):
        """
        where if categorical is None, columns from self.reject_null_good_of_fit will be ploted. Otherwise columns in cateigorical will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if categorical is None:
            categorical = self.reject_null_good_of_fit
            if not categorical:
                return print("There are not any non-uniform categorical variables stored in the model.\nEither none exist, or they haven't been fit.")

        self.univariate_categorical_snapshot(
                                        data=data,
                                        categorical=categorical,
                                        proportions=proportions,
                                        n_wide=n_wide,
                                        super_title=super_title)
        return
    
    #######################################################################################
    # A FUNCTION TO PLOT NUMERIC-TO-NUMERIC RELATIONSHIPS
    # PLOTS VALUES STORED IN self.above_threshold_corr_numnum
    #######################################################################################   

    def plot_bivariate_categoric_categoric(self,
                                           data:pd.DataFrame,  
                                           column_combinations:list|tuple|None=None,                      
                                        n_wide:int|tuple|list=(6,40,5),
                                        stacked_bars_when_max_bars_is_exceeded:bool=True,
                                        sorted:bool=False,
                                        super_title:str|None="Categoric-To-Categoric Bivariates With Significant Relationships"):
        """
        where if column_combinations is None, combinations from self.reject_null_catcat will be ploted. Otherwise combinations in column_combinations will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if column_combinations is None:
            column_combinations = self.reject_null_catcat
            if not column_combinations:
                return print("The model does not contain any categoric-to-categoric column pairs with significant relationships.\nEither none exist, or they haven't been fit.")
        self.bivaraite_categorical_snapshot(
                            data=data,
                            column_combinations=column_combinations,                        
                            n_wide=n_wide,
                            stacked_bars_when_max_bars_is_exceeded=stacked_bars_when_max_bars_is_exceeded,
                            sorted=sorted,
                            super_title=super_title)
        return
    
    #######################################################################################
    # A FUNCTION TO PLOT NUMERIC-TO-NUMERIC RELATIONSHIPS
    # PLOTS VALUES STORED IN self.above_threshold_corr_numnum
    #######################################################################################

    def plot_bivariate_numeric_numeric(self,
                                       data:pd.DataFrame,
                                       column_combos:list|tuple|None=None,
                                       plot_type:str='joint',
                                       linreg:bool=True,
                                       super_title:str='Numeric Bivariates With Significant Relationships',
                                       plot_type_kwargs:dict|None=None,
                                       linreg_kwargs:dict|None=None):
        """
        where if column_combos is None, combinations from self.above_threshold_corr_numnum will be ploted. Otherwise combos in column_combos will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if column_combos is None:
            column_combos = self.above_threshold_corr_numnum
            if not column_combos:
                return print("The model does not contain any numeric-to-numeric column pairs with significant relationships.\nEither none exist, or they haven't been fit.")
        self.bivariate_numeric_numeric_snapshot(
                                           data=data,
                                            column_combos=column_combos,
                                            plot_type=plot_type,
                                            linreg=linreg,                        
                                            super_title=super_title,
                                            plot_type_kwargs=plot_type_kwargs,
                                            linreg_kwargs=linreg_kwargs)
        return

    #######################################################################################
    # A FUNCTION TO PLOT NUMERIC-TO-CATEGORIC RELATIONSHIPS
    # PLOTS VALUES STORED IN self.reject_null_catnum  
    # where (num,cat) is the nested arrangement
    #######################################################################################

    def plot_numeric_to_categoric_relationships(self,
                                            data:pd.DataFrame,
                                            column_combos:list|tuple|None=None,
                                            plot_type:str='boxen', #box, boxen, or violin
                                            n_wide:int|tuple|list=(6,40,8),
                                            super_title:str|None='Numeric-to-Categoric Bivariates With Significant Relationships'):
        """
        where if column_combos is None, combinations from self.reject_null_catnum will be ploted. Otherwise combos in column_combos will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if column_combos is None:
            column_combos = self.reject_null_catnum 
            if not column_combos:
                return print("There are not any numeric-to-categoric column pairs with significant relationships.\nEither none exist, or they haven't been fit.")
        self.numeric_to_categorical_snapshot(data=data,
                                            column_combos=column_combos,
                                            plot_type=plot_type,
                                            n_wide=n_wide,
                                            super_title=super_title)
        return
    
    #######################################################################################
    # A FUNCTION TO PLOT SUPERCATEGORIES TO SUBCATEGORIES
    # PLOTS VALUES STORED IN self.supercategory_subcategory_pairs
    #######################################################################################

    def plot_super_subcats(self,
                        data:pd.DataFrame,
                        supercat_subcat_pairs:list|tuple|None=None,
                        row_height:int=2,
                        cols_per_row:int=3,
                        y_tick_fontsize:int=12
                        ):
        """
        where if supercat_subcat_pairs is None, combinations from self.supercategory_subcategory_pairs will be ploted. Otherwise combos in supercat_subcat_pairs will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if supercat_subcat_pairs is None:
            supercat_subcat_pairs = self.supercategory_subcategory_pairs
            if not supercat_subcat_pairs:
                return print("There are not any Supercategory-Subcategory relationships to plot.\nEither none exist, or they haven't been fit.")
        figure_map, figure_plot_params = self._prep_super_subcat_figure_maps(data, 
                                                supercat_subcat_pairs = supercat_subcat_pairs, 
                                                row_height=row_height, 
                                                cols_per_row=cols_per_row, 
                                                y_tick_fontsize=y_tick_fontsize
                                                )
        self.plot_supercats_subcats(data, 
                                figure_map, 
                                *figure_plot_params)
        return


    ############################################################################################
    # A WRAPPER FUNCTION THAT PLOTS ALL THE SUPPORTED PLOTS AVAILABLE
    # CALLS: self.plot_non_uniform_categorical(), self.plot_bivariate_categoric_categoric(), 
    #   self.plot_bivariate_numeric_numeric(), self.plot_numeric_to_categoric_relationships(), self.plot_super_subcats() 
    #   INTERNALLY
    ############################################################################################

    def produce_all_plots(self,
                  data:pd.DataFrame,
                  cat_univar:list|tuple|None|bool=None,
                  catcat_bivar:list|tuple|None|bool=None,
                  numnum_bivar:list|tuple|None|bool=None,
                  catnum_bivar:list|tuple|None|bool=None,  # where in practice num is placed before cat
                  super_subcat_pairs:list|tuple|None|bool=None,
                  #multivar not yet supported,
                  cat_univar_params:dict={
                                        'proportions':False,
                                        'n_wide':(6,40,4),
                                        'super_title':"Univariate Categorical Variables - Reject Good-Of-Fit for Uniform"},
                  catcat_bivar_params:dict={
                                        'n_wide':(6,40,5),
                                        'stacked_bars_when_max_bars_is_exceeded':True,
                                        'sorted':False,
                                        'super_title':"Categoric-To-Categoric Bivariates With Significant Relationships"},
                  numnum_bivar_params:dict={
                                        'plot_type':'joint',
                                        'linreg':True,
                                        'super_title':'Numeric Bivariates With Significant Relationships',
                                        'plot_type_kwargs':None,
                                        'linreg_kwargs':None},
                  catnum_bivar_params:dict={
                                        'plot_type':'boxen', #box, boxen, or violin
                                        'n_wide':(6,40,8),
                                        'super_title':'Numeric-to-Categoric Bivariates With Significant Relationships'},
                  super_subcat_pairs_params:dict={                      
                                        'row_height':2,
                                        'cols_per_row':3,
                                        'y_tick_fontsize':12}     ): 
        """
        where data is the dataframe values are taken from
        cat_univar, catcat_bivar, numnum_bivar, catnum_bivar, and super_subcat_pairs
            can be of list|tuple|None|bool. 
                default is None
                if None, then values are taken from class objects stored in fit calls
                if False, then values are not ploted
        cat_univar_params, catcat_bivar_params, numnum_bivar_params, catnum_bivar_params, super_subcat_pairs_params
            accept custom plot parameters
        """
        if cat_univar!=False:
            self.plot_non_uniform_categorical(
                                            data=data,
                                            categorical=cat_univar,
                                            **cat_univar_params)
        else:
            print('Plot Univariate is set to False')
        if catcat_bivar!=False:   
            self.plot_bivariate_categoric_categoric(
                                            data=data,  
                                            column_combinations=catcat_bivar,
                                            **catcat_bivar_params) 
        else:
            print('Plot Bivariate Categoric-Categoric is set to False') 
        if numnum_bivar!=False:
            self.plot_bivariate_numeric_numeric(
                                            data=data,
                                            column_combos=numnum_bivar,
                                            **numnum_bivar_params)
        else:
            print('Plot Bivariate Numeric-Numeric is set to False')
        if catnum_bivar!=False:
            self.plot_numeric_to_categoric_relationships(
                                            data=data,
                                            column_combos=catnum_bivar,
                                            **catnum_bivar_params)
        else:
            print('Plot Bivariate Numeric-Categoric is set to False')
        if super_subcat_pairs!=False:
            self.plot_super_subcats(
                                            data=data,
                                            supercat_subcat_pairs=super_subcat_pairs,
                                            **super_subcat_pairs_params)
        else:
            print('Plot Supercategory-Subcategory Partitions is set to False')
        return
    #######################################################################################
    # FUNCTIONS AND HELPER FUNCITONS TO PLOT COLUMN(S) AND RELATIONSHIPS INVOLVING A TARGET OR TARGETS
    # PLOTS VALUES STORED IN self.target_cols_with_significant_matches_and_not_significant
    #######################################################################################


    def _prepare_target_plot_data(self,
                                 target_column:str,
                                 reject_catnum:bool=True,
                                 reject_numnum:bool=True,
                                 reject_catcat:bool=True,
                                 is_super_or_subcat:bool=True,
                                 not_uniform_or_reject_normal:bool=True,
                                 reject_multivariates:bool=False):    #FALSE BECAUSE MULITVARIATE VISULIZATIONS ARE NOT YET SUPPORTED
        """
        where target_column is a string and other parameters are T/F to indicate whether or not they should be included
        """
        
        meta_dict_ = self.target_cols_with_significant_matches_and_not_significant.get(target_column,None)
        # create a copy to allow use modifications to meta_dict w/o affecting class the object
        if meta_dict_!=None:
            meta_dict=meta_dict_.copy()
        if meta_dict is None:
            if isinstance(target_column,str):
                raise ValueError(f"{target_column} has not been fit.")
            else:
                raise ValueError(f"Expected data type == string, found {type(target_column)}")
            
        catnum, numnum, catcat, supersubcat, univariate, multivariate = [],[],[],[],[],[]

        data_type = self.target_cols_with_significant_matches_and_not_significant[target_column]['target_dtype'][0]

        # match target with numeric cols
        if meta_dict['significant_numeric_relationships']:            
            nums=  zip(meta_dict['significant_numeric_relationships'], meta_dict['significant_numeric_tests'])
            if data_type=='numeric' and reject_numnum==True:
                for col, test in nums:
                    result=[target_column,col,test]
                    numnum.append(result)
            elif data_type=='categoric' and reject_catnum==True:
                for col, test in nums:
                    result=[col,target_column,test]
                    catnum.append(result)  # where catnum is not the order stored anywhere else either. stored are [num, cat]
        # match target with categoric cols
        if meta_dict['significant_categoric_relationships']:
            cats=  zip(meta_dict['significant_categoric_relationships'], meta_dict['significant_categoric_tests'])
            if data_type=='numeric' and reject_catnum==True:
                for col, test in cats:
                    result=[target_column,col,test]
                    catnum.append(result)
            elif data_type=='categoric' and reject_catcat==True:
                for col, test in cats:
                    result=[col,target_column,test]
                    catcat.append(result) 
        # match with multivarieate combos
        if meta_dict['significant_categoric_combination_group_relationship'] and reject_multivariates==True:
            combs= zip(meta_dict['significant_categoric_combination_group_relationship'], meta_dict['significant_categoric_combination_group_relationship_test_type'])
            if data_type=='numeric':
                for col, test in combs:
                    result=[[target_column,'numeric'],col,test]
                    multivariate.append(result)
            elif data_type=='categoric':
                for col, test in combs:
                    result=[[target_column,'categoric'],col,test]
                    multivariate.append(result) 
        # match to super or subcat if exists
        if meta_dict['paired_to_a_supercategory'] and is_super_or_subcat==True:
            for match in meta_dict['paired_to_a_supercategory']:
                res = [match, target_column]
                supersubcat.append(res)
        if meta_dict['paired_to_a_subcategory'] and is_super_or_subcat==True:
            for match in meta_dict['paired_to_a_subcategory']:
                res = [target_column, match]
                supersubcat.append(res)
        # if it's a univariate to plot
        plottable_responses = ('fail_to_reject_uniform')  # potential to have normal distribution support too, but not supported at this time
        if (not_uniform_or_reject_normal==True) and any( result in plottable_responses for result in meta_dict['is_normal_or_uniform'] ):
               univariate.append(target_column)
        return  catnum, numnum, catcat, supersubcat, univariate, multivariate
    

    def _fit_target_visualizations(self,
                    data:pd.DataFrame,
                    targets:list|tuple|str,
                    reject_catnum:bool=True,
                    reject_numnum:bool=True,
                    reject_catcat:bool=True,
                    is_super_or_subcat:bool=True,
                    not_uniform_or_reject_normal:bool=True,
                    reject_multivariates:bool=False,
                    auto_fit:bool=True,   # to call fit function(s) when needed
                    ): 
        
        """
        where targets is list|tuple of targets to plot. string is accepted too in cases of one target
        auto_fit==True indicates the target should be fit if not already, otherwise a RuntimeError will raise
        all other parameters are bool T/F to indicate whether they should be included
        """
        # ensure listlike targets
        if isinstance(targets,str):
            targets=[targets]
        # create a template dict
        def target_dict_template():
            return {'reject_catnum':[],
                    'reject_numnum':[],
                    'reject_catcat':[],
                    'is_super_or_subcat':[],
                    'not_uniform_or_reject_normal':[],
                    'reject_multivariates':[]
                    }
        # this will be updated based on target_dict_template() for each target
        targets_and_results = {}
        # loop through targets
        for target in targets:
            # determine target datatype
            target_dtype=data[target].dtype
            # check if the target has been fit at all (such as hyp test w other columns), and if auto_fit==True, call fit if needed
            if target not in self.has_called_fit_column_relationships:
                if auto_fit==True:
                    if target_dtype not in ('object','category'):
                        numeric_target=target
                        categoric_target=None
                    else:
                        numeric_target=None
                        categoric_target=target
                    self.fit_column_relationships(df=data,
                                 numeric_target=numeric_target,
                                 categoric_target=categoric_target,
                                 )
                else:
                    raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_column_relationships() needs to be called.')                
            # check if the target has been fit with multivariates, and if auto_fit==True, call fit if needed
            if (reject_multivariates==True) and (target not in self.has_called_fit_multivariate_column_relationships):
                if auto_fit==True:
                    multivariate_params= self.multivariate_params
                    self.fit_multivariate_column_relationships(df=data,
                                              targets=target,
                                              **multivariate_params,
                                              )
                else:
                    raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_multivariate_column_relationships() needs to be called.')
            # check if the target has been tested/fit for to a uniform distribution, and if auto_fit==True, call fit if needed            
            if (not_uniform_or_reject_normal==True) and (target_dtype in ('object','category')) and (target not in self.has_called_fit_goodness_of_fit_uniform):
                if auto_fit==True:
                    self.fit_goodness_of_fit_uniform(data,
                                    categoric_columns=target)
                else:
                    raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_goodness_of_fit_uniform() needs to be called.')
            # check if the target has been tested/fit for super_subcat relationships, and if auto_fit==True, call fit if needed           
            if (is_super_or_subcat==True) and (target_dtype in ('object','category')) and (target not in self.has_called_fit_supercat_subcat_pairs):
                if auto_fit==True:
                    pairs_list=[[i,target] for i in data.select_dtypes(['object','category']).columns if i != target]
                    if pairs_list:
                        supsub_params=self.supercat_subcat_params 
                        self.fit_supercat_subcat_pairs(data,
                                                    **supsub_params,
                                                        pairs_list=pairs_list )                
                else:
                    raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_supercat_subcat_pairs() needs to be called.')

            one_targ_catnum, one_targ_numnum, one_targ_catcat, one_targ_supersubcat, one_targ_univariate, one_targ_multivariate = self._prepare_target_plot_data(
                                                                            target_column=target,
                                                                            reject_catnum=reject_catnum,
                                                                            reject_numnum=reject_numnum,
                                                                            reject_catcat=reject_catcat,
                                                                            is_super_or_subcat=is_super_or_subcat,
                                                                            not_uniform_or_reject_normal=not_uniform_or_reject_normal,
                                                                            reject_multivariates=reject_multivariates)
            
            targets_and_results[target]=target_dict_template()
            targets_and_results[target]['reject_catnum']=targets_and_results[target]['reject_catnum']+one_targ_catnum
            targets_and_results[target]['reject_numnum']=targets_and_results[target]['reject_numnum']+one_targ_numnum
            targets_and_results[target]['reject_catcat']=targets_and_results[target]['reject_catcat']+one_targ_catcat
            targets_and_results[target]['is_super_or_subcat']=targets_and_results[target]['is_super_or_subcat']+one_targ_supersubcat
            targets_and_results[target]['not_uniform_or_reject_normal']=targets_and_results[target]['not_uniform_or_reject_normal']+one_targ_univariate
            targets_and_results[target]['reject_multivariates']=targets_and_results[target]['reject_multivariates']+one_targ_multivariate
            
        return targets_and_results


    def visualize_by_targets(self,
                    data:pd.DataFrame,
                    targets:list|tuple|str,
                    reject_catnum:bool=True,
                    reject_numnum:bool=True,
                    reject_catcat:bool=True,
                    is_super_or_subcat:bool=True,
                    not_uniform_or_reject_normal:bool=True,
                    reject_multivariates:bool=False,
                    auto_fit:bool=True,   # to call fit function(s) when needed
                    targets_share_plots:bool=False
                    ):
        """
        where data is the dataframe that holds values
        targets is a string target variable or list|tuple of target(s)
        auto_fit indicates whethere to fit variables that haven't been fit, or to raise a RuntimeError
        targets_share_plots indicates whether to put all targets on the same figures, or to create seperate sets of figures for each target
        other parameters are True/False bool to indicate whether to include in plots. 
            they are:
                reject_catnum: significant categorical-numerical relationship pairs
                reject_numnum: significant numerical-numerical relationship pairs
                reject_catcat: significant categorical-categorical relationship pairs
                is_super_or_subcat: supercategory-subcategory pairs
                not_uniform_or_reject_normal: categorical variables that don't follow a uniform distribution {rejected normal distribution not yet supported for numeric},
                reject_multivariates: where target columns are tested/compared to concatenated variables
        """
        ## should be made into a class object and updated here such as [targ for targ in targets if targ not in self.target_dict.keys()) self.targets_dict.update(result)
        targets_dict =  self._fit_target_visualizations(
                    data=data,
                    targets=targets,
                    reject_catnum=reject_catnum,
                    reject_numnum=reject_numnum,
                    reject_catcat=reject_catcat,
                    is_super_or_subcat=is_super_or_subcat,
                    not_uniform_or_reject_normal=not_uniform_or_reject_normal,
                    reject_multivariates=reject_multivariates,
                    auto_fit=auto_fit,   # to call fit function(s) when needed
                    )
        # plot targets on one plot per plot type
        if targets_share_plots==True:
            reject_catnum, reject_numnum, reject_catcat, is_super_or_subcat, not_uniform_or_reject_normal, reject_multivariates=[],[],[],[],[],[]
            #iterate through dict and make sure no combos are repeated
            for k,v in targets_dict.items():

                #no repeat reject catnum
                if reject_catnum:
                    vrcn=[]
                    for val in v['reject_catnum']:
                        # determine if val is already in reject_catnum or not
                        is_new=True
                        for first_val in reject_catnum:
                            if ((val[0]==first_val[0]) and (val[1]==first_val[1])) or ((val[0]==first_val[1]) and (val[1]==first_val[0])):
                                is_new=False
                                break
                        if is_new==True:
                            vrcn.append(val)
                    reject_catnum=reject_catnum+vrcn
                else:
                    reject_catnum = v['reject_catnum']

                # no repeat numnum
                if reject_numnum:
                    vrnn=[]
                    for val in v['reject_numnum']:
                        # determine if val is already in reject_numnum or not
                        is_new=True
                        for first_val in reject_numnum:
                            if ((val[0]==first_val[0]) and (val[1]==first_val[1])) or ((val[0]==first_val[1]) and (val[1]==first_val[0])):
                                is_new=False
                                break
                        if is_new==True:
                            vrnn.append(val)
                    reject_numnum=reject_numnum+vrnn
                else:
                    reject_numnum=v['reject_numnum']

                # no repeat catcat
                if reject_catcat:
                    vrcc=[]
                    for val in v['reject_catcat']:
                        is_new=True
                        for first_val in reject_catcat:
                            if ((val[0]==first_val[0]) and (val[1]==first_val[1])) or ((val[0]==first_val[1]) and (val[1]==first_val[0])):
                                is_new=False
                                break
                        if is_new==True:
                            vrcc.append(val)
                    reject_catcat=reject_catcat+vrcc
                else:
                    reject_catcat = v['reject_catcat']

                #no repeat super subcat combos
                if is_super_or_subcat:
                    visos=[]
                    for val in v['is_super_or_subcat']:
                        is_new=True
                        for first_val in is_super_or_subcat:
                            if ((val[0]==first_val[0]) and (val[1]==first_val[1])) or ((val[0]==first_val[1]) and (val[1]==first_val[0])):
                                is_new=False
                                break
                        if is_new==True:
                            visos.append(val)
                    is_super_or_subcat=is_super_or_subcat+visos 
                else:
                    is_super_or_subcat=v['is_super_or_subcat']

                # no need to filter duplicates for good of fit uniform
                not_uniform_or_reject_normal=not_uniform_or_reject_normal+v['not_uniform_or_reject_normal']
                
                # there may be duplicates in multivariate, but because of the 'target' approach they are not filtered out of plots ==>> (ie target on one axis, other vars on the other)
                reject_multivariates=reject_multivariates+v['reject_multivariates']
            
            # don't pass empty lists or tuples
            if not reject_catnum: reject_catnum=False
            if not reject_numnum: reject_numnum=False
            if not reject_catcat: reject_catcat=False
            if not is_super_or_subcat: is_super_or_subcat=False
            if not not_uniform_or_reject_normal: not_uniform_or_reject_normal=False
            if not reject_multivariates: reject_multivariates=False
            # plot vars
            print(f"SIGNIFICANT VISUALIZATIONS FOR VARAIBLES IN:\n          {targets}")
            print('Non-Value Plots will Automatically be set to False')
            self.produce_all_plots(
                                data=data,
                                cat_univar=not_uniform_or_reject_normal,
                                catcat_bivar=reject_catcat,
                                numnum_bivar=reject_numnum,
                                catnum_bivar=reject_catnum,
                                super_subcat_pairs=is_super_or_subcat)
        # plot targets individually
        else:
            for k,v in targets_dict.items():
                reject_catnum=v['reject_catnum']
                reject_numnum=v['reject_numnum']
                reject_catcat=v['reject_catcat']
                is_super_or_subcat=v['is_super_or_subcat']
                not_uniform_or_reject_normal=v['not_uniform_or_reject_normal']
                reject_multivariates=v['reject_multivariates']
                # don't pass empty lists or tuples
                if not reject_catnum: reject_catnum=False
                if not reject_numnum: reject_numnum=False
                if not reject_catcat: reject_catcat=False
                if not is_super_or_subcat: is_super_or_subcat=False
                if not not_uniform_or_reject_normal: not_uniform_or_reject_normal=False
                if not reject_multivariates: reject_multivariates=False
                # call the plot function
                print('= = = = = '*20)
                print(f"SIGNIFICANT VISUALIZATIONS FOR {k}")
                print('Non-Value Plots will Automatically be set to False')
                self.produce_all_plots(
                                    data=data,
                                    cat_univar=not_uniform_or_reject_normal,
                                    catcat_bivar=reject_catcat,
                                    numnum_bivar=reject_numnum,
                                    catnum_bivar=reject_catnum,
                                    super_subcat_pairs=is_super_or_subcat)
                print('= = = = = '*20)
        return
