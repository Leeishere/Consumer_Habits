import numpy as np
import pandas as pd
import streamlit as st

from utils.AnalyzeDataset import AnalyzeDataset

# ======================================|
# FLOW CONTROL VARIABLES                |
# ======================================|
# st.session_state.preprocessing_done   |
# st.session_state.early_stop           |
# st.session_state.drops_complete       |
# st.session_state.types_confirmed      |
# st.session_state.dates_pieced         |
# st.session_state.is_ready_for_fit     |
# st.session_state.fit                  |
# --------------------------------------|



if 'n_wide' not in st.session_state:
    st.session_state.n_wide = {'n_wide': [8, 30, 4]}

if "cat_univar_params" not in st.session_state:
    st.session_state.cat_univar_params = {"proportions": False, "n_wide": (8, 30, 4), "super_title": "Univariate Categorical Variables - Reject Good-Of-Fit for Uniform"}

if "catcat_bivar_params" not in st.session_state:
    st.session_state.catcat_bivar_params = {"n_wide": (8, 30, 4), "stacked_bars_when_max_bars_is_exceeded": True, "sorted": False, "super_title": "Categoric-To-Categoric Bivariates - Reject Null"}

if "numnum_bivar_params" not in st.session_state:
    st.session_state.numnum_bivar_params = {"plot_type": "joint", "linreg": False, "plot_type_kwargs": None, "linreg_kwargs": None, "super_title": "Numeric Bivariates With Significant Correlation(s)"}

if "numcat_bivar_params" not in st.session_state:
    st.session_state.numcat_bivar_params = {"plot_type": "boxen", "n_wide": (8, 30, 4), "super_title": "Numeric-to-Categoric Bivariates  - Reject Null"}

if "super_subcat_pairs_params" not in st.session_state:
    st.session_state.super_subcat_pairs_params = {"row_height": 3, "cols_per_row": 2, "y_tick_fontsize": 12, "super_title": "Supercategory-Subcategory - One Categoric Variable Partitions Another"}

if "num_univar_params" not in st.session_state:
    st.session_state.num_univar_params = {"kde": None, "proportions": False, "n_wide": (8, 30, 4), "super_title": "Univariate Numerical Variables - Reject Normal Distribution", "force_significant_bin_edges": None, "minimize_significant_bins": None, "include_multivariate": True}

if "kruskal_assumption_check_params" not in st.session_state:
    st.session_state.kruskal_assumption_check_params = {"levene_alpha": 0.01, "ks_alpha": 0.01, "return_pseudo": True, "pseudo_test_max_global_ties_ratio": 0.7, "full_pseudo": False, "dropna": True, "n_jobs": 4, "guesstimate": {"rej_max_pct_in_group": 0.2, "max_num_outlier_all_reject": 3, "max_pct_reject_total": 0.2}}

if "anova_assumption_check_params" not in st.session_state:
    st.session_state.anova_assumption_check_params = {"normality_alpha": 0.01, "homogeneity_alpha": 0.01, "min_n": 5, "iqr_multiplier": 2, "dropna": True}

if "chi2_assumption_check_params" not in st.session_state:
    st.session_state.chi2_assumption_check_params = {"dropna": True}

if "supercat_subcat_params" not in st.session_state: 
    st.session_state.supercat_subcat_params = {"max_evidence": 0.2, "isolate_super_subs": False}

if "multivariate_params" not in st.session_state:
    st.session_state.multivariate_params = {"max_n_combination_size": 2, "max_n_combinations": 20_000, "min_combo_size": 2}

if "multivariate_concatenation_delimiter" not in st.session_state:
    st.session_state.multivariate_concatenation_delimiter = "_|&|_"

if "numnum_meth_alpha_above_instructions" not in st.session_state:
    st.session_state.numnum_meth_alpha_above_instructions = [["pearson", 0.6, None], ["spearman", 0.6, None], ["kendall", 0.6, None]]

if "numcat_meth_alpha_above_instructions" not in st.session_state:
    st.session_state.numcat_meth_alpha_above_instructions = [["kruskal", 0.05, None], ["anova", 0.05, None]]

if "catcat_meth_alpha_above_instructions" not in st.session_state:
    st.session_state.catcat_meth_alpha_above_instructions = [["chi2", 0.05, None]]

if "good_of_fit_uniform_test_instrucions" not in st.session_state:
    st.session_state.good_of_fit_uniform_test_instrucions = [0.05, None]

if "normal_test_instructions" not in st.session_state:
    st.session_state.normal_test_instructions = [0.05, None]

if "datatypes_df" not in st.session_state:
    st.session_state.datatypes_df = None

if 'result_df' not in st.session_state:
    st.session_state.result_df = None

if 'data' not in st.session_state:
    st.session_state.data = None

if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

if 'user_choice_datatype_changes' not in st.session_state:
    st.session_state.user_choice_datatype_changes = None

if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []

if 'native_date_cols' not in st.session_state:
    st.session_state.native_date_cols = []

if 'over_unique_threshold' not in st.session_state:
    st.session_state.over_unique_threshold = []

if 'user_drop_instrucitons' not in st.session_state:
    st.session_state.user_drop_instrucitons = None

if 'drops_complete' not in st.session_state:
    st.session_state.drops_complete = False

if 'types_confirmed' not in st.session_state:
    st.session_state.types_confirmed = False

if 'dates_pieced' not in st.session_state:
    st.session_state.dates_pieced = False

if "page" not in st.session_state:
    st.session_state.page = "Data Upload & Processing"

if "is_ready_for_fit" not in st.session_state:
    st.session_state.is_ready_for_fit = False

if "fit" not in st.session_state:
    st.session_state.fit = False

if "preprocessing_done" not in st.session_state:
    st.session_state.preprocessing_done = False

if "original_len" not in st.session_state:
    st.session_state.original_len = None

if "original_shape" not in st.session_state:
    st.session_state.original_shape = None

if 'max_date_cols_used' not in st.session_state:
    st.session_state.max_date_cols_used = None

if 'min_pct_non_null_to_propose_a_dtype' not in st.session_state:
    st.session_state.min_pct_non_null_to_propose_a_dtype = None

if 'max_unique_pct_of_total_ie_identifier_variable_ID' not in st.session_state:
    st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID = None

if 'max_pct_unique_for_numtype_cattype_threshold' not in st.session_state:
    st.session_state.max_pct_unique_for_numtype_cattype_threshold = None

if 'early_stop' not in st.session_state:
    st.session_state.early_stop = None
    
def set_page():
    st.session_state.page = st.session_state.navigation_control





if st.session_state.page == "Data Upload & Processing":
    

    # Sidebar for threshold adjustments
    with st.sidebar:

        # Page navigation #
        st.header("Navigation")        
        st.session_state.page = st.radio("Navigate", 
                                         ["Data Upload & Processing", "Group Visualizations", "Target Visualizations"], 
                                         index=["Data Upload & Processing", "Group Visualizations", "Target Visualizations"].index(st.session_state.page),
                                         key = 'navigation_control',
                                         on_change = set_page)

        st.header("Threshold Adjustments")
        
        # Correlation thresholds for numeric-numeric
        current_corr = st.session_state.numnum_meth_alpha_above_instructions[0][1]
        corr_options = [current_corr,0.7,0.8,0.9]
        corr_labels = [f"{current_corr}"] + [str(i) for i in corr_options[1:]]
        selected_corr = st.selectbox("Correlation Threshold (Numeric-Numeric)", corr_labels, index=0)
        if selected_corr != corr_labels[0]:
            new_val = corr_options[corr_labels.index(selected_corr)]
            for item in st.session_state.numnum_meth_alpha_above_instructions:
                item[1] = new_val
        
        
        # P-value threshold for numeric-categoric
        current_p_numcat = st.session_state.numcat_meth_alpha_above_instructions[0][1]
        p_numcat_options = [current_p_numcat, 0.025, 0.01]
        p_numcat_labels = [f"{current_p_numcat}"] + [str(i) for i in p_numcat_options[1:]]
        selected_p_numcat = st.selectbox("P-value Threshold (Numeric-Categoric)", p_numcat_labels, index=0)
        if selected_p_numcat != p_numcat_labels[0]:
            new_val = p_numcat_options[p_numcat_labels.index(selected_p_numcat)]
            for item in st.session_state.numcat_meth_alpha_above_instructions:
                item[1] = new_val
        
        # P-value threshold for categoric-categoric
        current_p_catcat = st.session_state.catcat_meth_alpha_above_instructions[0][1]
        p_catcat_options = [current_p_catcat, 0.025, 0.01]
        p_catcat_labels = [f"{current_p_catcat}"] + [str(i) for i in p_catcat_options[1:]]
        selected_p_catcat = st.selectbox("P-value Threshold (Categoric-Categoric)", p_catcat_labels, index=0)
        if selected_p_catcat != p_catcat_labels[0]:
            new_val = p_catcat_options[p_catcat_labels.index(selected_p_catcat)]
            st.session_state.catcat_meth_alpha_above_instructions[0][1] = new_val
        
        # P-value threshold for good-of-fit uniform
        current_p_gof = st.session_state.good_of_fit_uniform_test_instrucions[0]
        p_gof_options = [current_p_gof, 0.025, 0.01]
        p_gof_labels = [f"{current_p_gof}"] + [str(i) for i in p_gof_options[1:]]
        selected_p_gof = st.selectbox("P-value Threshold (Good-of-Fit Uniform)", p_gof_labels, index=0)
        if selected_p_gof != p_gof_labels[0]:
            new_val = p_gof_options[p_gof_labels.index(selected_p_gof)]
            st.session_state.good_of_fit_uniform_test_instrucions[0] = new_val
        
        # P-value threshold for normal test
        current_p_norm = st.session_state.normal_test_instructions[0]
        p_norm_options = [current_p_norm, 0.025, 0.01]
        p_norm_labels = [f"{current_p_norm}"] + [str(i) for i in p_norm_options[1:]]
        selected_p_norm = st.selectbox("P-value Threshold (Normal Test)", p_norm_labels, index=0)
        if selected_p_norm != p_norm_labels[0]:
            new_val = p_norm_options[p_norm_labels.index(selected_p_norm)]
            st.session_state.normal_test_instructions[0] = new_val

        # decide to isolate partitioning super-subcat pairs
        curr_isolate = st.session_state.supercat_subcat_params["isolate_super_subs"]
        isolate_options = [curr_isolate, not curr_isolate]
        selected_isolate = st.selectbox("Isolate Partitioning Super-Subcategory Counterparts",isolate_options,index=0)
        if selected_isolate != st.session_state.supercat_subcat_params['isolate_super_subs']:
            st.session_state.supercat_subcat_params.update({'isolate_super_subs':selected_isolate })

        
        # return pseudo{'full_pseudo':False}
        curr_ret_p = st.session_state.kruskal_assumption_check_params['return_pseudo']
        retp_options = [curr_ret_p, not curr_ret_p]
        selected_retp = st.selectbox("Test Similarity Instead When Strict Mean Assumptions Aren't Met\nIn Kruskal-Wallis Test",retp_options,index=0)
        if selected_retp != st.session_state.kruskal_assumption_check_params['return_pseudo']:
            st.session_state.kruskal_assumption_check_params.update({'return_pseudo':selected_retp })

        # return pseudo
        curr_ps = st.session_state.kruskal_assumption_check_params['full_pseudo']
        ps_options = [curr_ps, not curr_ps]
        selected_ps = st.selectbox("Test Distribution Similarity Instead of Mean\nIn Kruskal-Wallis Test",ps_options,index=0)
        if selected_ps != st.session_state.kruskal_assumption_check_params['full_pseudo']:
            st.session_state.kruskal_assumption_check_params.update({'full_pseudo':selected_ps})




    #=================================================================================================================================================================

    # CSV File Upload and Processing
    st.title("Data Analyzer App")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="File size limit: 10MB")

    if uploaded_file is not None:

        #   multipliers to help name a dtype
        st.session_state.min_pct_non_null_to_propose_a_dtype                        = 0.99
        st.session_state.max_pct_unique_for_numtype_cattype_threshold               = 0.005 # over for num <= for cat
        st.session_state.max_date_cols_used                                         = 5
        st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID          = 0.4  # percentage of all observations that are unique to a columns, such as ID number or unique locations



        # Check file size (10MB limit for fly.io)
        file_size = len(uploaded_file.getvalue())
        if file_size > 10 * 1024 * 1024:
            st.error("File size exceeds 10MB limit. Please upload a smaller file.")
        else:

            try:

                # Read CSV to DataFrame
                # Only re-read CSV when a new file is uploaded; preserve session state data across reruns
                if st.session_state.uploaded_file_name != uploaded_file.name:
                    st.session_state.drops_complete, st.session_state.types_confirmed, st.session_state.dates_pieced = False, False, False
                    st.session_state.fit, st.session_state.preprocessing_done, st.session_state.is_ready_for_fit     = False, False, False
                    st.session_state.early_stop = False

                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.data = pd.read_csv(uploaded_file)   
                    if len(st.session_state.data.shape)<2:
                        st.session_state.data = st.session_state.data.to_frame()
                    # capture original len and shape
                    st.session_state.original_len = len(st.session_state.data)
                    st.session_state.original_shape = st.session_state.data.shape

                    # display a sample
                    st.markdown('5 sample rows of existing values')
                    st.dataframe(st.session_state.data.sample(min(5, len(st.session_state.data))))

                if  (  (not st.session_state.early_stop) and (st.session_state.drops_complete == False) and (st.session_state.types_confirmed == False) and (st.session_state.dates_pieced == False) and (st.session_state.fit == False) and (st.session_state.preprocessing_done == False) and (st.session_state.is_ready_for_fit == False) ):

                    # Drop nulls
                    # DROPNA NOT REQUIRED. MODULE WILL HANDLE NAN INTERNALLY
                    #st.session_state.data = st.session_state.data.dropna()

                    new_len = len(st.session_state.data)

                    # store cols and meta
                    cols_and_meta = {}
                    # Robust datatype detection
                    date_columns = []
                    native_date_cols = []
                    over_unique_threshold = []
                    for col in st.session_state.data.columns:
                        proposed_dtype = None
                        col_dtype = st.session_state.data[col].dtype
                        niq = st.session_state.data[col].nunique()

                        # check unique to capture identifier columns
                        if niq > (new_len*st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID):
                            over_unique_threshold.append(col)

                        if str(col_dtype).startswith('datetime') or str(col_dtype).startswith('timedelta'):
                            date_columns.append(col)
                            native_date_cols.append(col)
                            proposed_dtype = 'datetime'
                        # Try to detect dates/datetimes
                        elif str(col_dtype) in ('object','category'):
                            test_as_dt = pd.to_datetime(st.session_state.data[col].copy(), errors='coerce').notna().sum()

                            # if st.session_state.min_pct_non_null_to_propose_a_dtype% convert to datetime it can be datetime
                            if (test_as_dt >= (st.session_state.min_pct_non_null_to_propose_a_dtype * new_len)):
                                date_columns.append(col)
                                proposed_dtype = 'datetime'

                            else:
                                # attempt to parse as numeric
                                test_as_num = pd.to_numeric(st.session_state.data[col].copy(), errors='coerce').notna().sum()

                                # if >st.session_state.min_pct_non_null_to_propose_a_dtype% are valid values and over st.session_state.max_pct_unique_for_numtype_cattype_threshold are unique, it can be int or float
                                if (test_as_num>(new_len*st.session_state.min_pct_non_null_to_propose_a_dtype)) and (niq>(new_len*st.session_state.max_pct_unique_for_numtype_cattype_threshold)):
                                    if abs( (st.session_state.data[col].copy().astype(float) - st.session_state.data[col].copy().astype(int)).sum() )>0:
                                        proposed_dtype = 'float64'
                                    else:
                                        proposed_dtype = 'int64'
                                else:
                                    proposed_dtype = 'category'

                        elif str(col_dtype).startswith('int') or str(col_dtype).startswith('float') or str(col_dtype).startswith('UInt'):
                            # if not over st.session_state.max_pct_unique_for_numtype_cattype_threshold are unique, it will be categoric
                            if (niq>(new_len*st.session_state.max_pct_unique_for_numtype_cattype_threshold)):
                                if (st.session_state.data[col].copy().astype(float) - st.session_state.data[col].copy().astype(int)).sum()>0:
                                    proposed_dtype = 'float64'
                                else:
                                    proposed_dtype = 'int64'
                            else:
                                proposed_dtype = 'category'

                        cols_and_meta[col] = [col_dtype,
                                            proposed_dtype,
                                            niq,
                                            False]

                    index = ['original_dtype', 'possible_change', 'n_unique','accept_change']
                    st.session_state.datatypes_df = pd.DataFrame(cols_and_meta, index=index)
                    st.session_state.date_columns = date_columns
                    st.session_state.native_date_cols = native_date_cols
                    st.session_state.over_unique_threshold = over_unique_threshold

                    st.session_state.preprocessing_done = True

                if  (  (not st.session_state.early_stop) and (st.session_state.drops_complete == False) and (st.session_state.types_confirmed == False) and (st.session_state.dates_pieced == False) and (st.session_state.fit == False) and (st.session_state.preprocessing_done == True) and (st.session_state.is_ready_for_fit == False) ):

                    # Restore detection results from session state on reruns
                    new_len = len(st.session_state.data)
                    date_columns = st.session_state.date_columns
                    native_date_cols = st.session_state.native_date_cols
                    over_unique_threshold = st.session_state.over_unique_threshold
                    # display datatypes as df

                    transition_df = st.session_state.datatypes_df.copy().T
                    id_mask   = (transition_df['n_unique']>1)
                    auto_drop = transition_df.loc[~id_mask].index  
                    st.header("Drop Columns")
                    if len(auto_drop)>0:
                        st.info(f"Some columns have only one unique value and will be dropped by default:  {[i for i in auto_drop]}")
                        transition_df = transition_df.loc[id_mask]
                    # capture identifier variables now 
                    identifier_df = transition_df.loc[[i for i in over_unique_threshold if i not in auto_drop],['original_dtype', 'possible_change', 'n_unique']]
                    identifier_df['pct_unique'] = identifier_df['n_unique']/new_len
                    id_orig_str, id_poss_str = identifier_df['original_dtype'].astype(str), identifier_df['possible_change'].astype(str)
                    identifier_df['to_drop'] = ((id_orig_str.isin(['category','object']) & id_poss_str.isin(['category','object'])) )
                    identifier_df['pct_unique'] = round((identifier_df['pct_unique']*100).astype(float),2)
                    identifier_df = identifier_df[['original_dtype', 'possible_change', 'pct_unique', 'to_drop']]
                    # engage user
                    st.markdown("For Best Results, Drop Columns That Have a High Percentage of Unique or Distinct Values. Such as Identifier ID's")
                    drop_col_left, drop_col_right = st.columns([2, 1])
                    with drop_col_left:
                        st.session_state.user_drop_instrucitons = st.data_editor(identifier_df)
                    with drop_col_right:
                        st.markdown("**Also drop any additional columns:**")
                        extra_cols_to_drop = st.multiselect(
                            "Select columns to drop",
                            options=list(i for i in st.session_state.data.columns if i not in auto_drop),
                            default=[]
                        )
                    process_drop = st.button("Drop Columns")

                    #process user instrucions
                    cols_to_drop = []
                    if process_drop:
                        cols_to_drop = list(st.session_state.user_drop_instrucitons.loc[st.session_state.user_drop_instrucitons['to_drop']==True].index)
                        cols_to_drop = list(set(cols_to_drop + list(extra_cols_to_drop) + list(auto_drop)))
                        st.session_state.data = st.session_state.data.drop(columns=cols_to_drop, errors='ignore')
                        if len(st.session_state.data.shape)<2:
                            st.session_state.data = st.session_state.data.to_frame()
                        st.session_state.datatypes_df = st.session_state.datatypes_df.drop(columns=cols_to_drop, errors='ignore')
                        if 0 < len(st.session_state.data.shape) < 2:
                            st.session_state.data = st.session_state.data.to_frame()
                        if st.session_state.data.shape[1] == 0:
                            st.info("All Columns Have Been Dropped. There is No Data to Process.")
                            st.session_state.early_stop = True
                        
                        st.session_state.drops_complete = True

                if  ( (not st.session_state.early_stop) and (st.session_state.drops_complete == True) and (st.session_state.types_confirmed == False) and (st.session_state.dates_pieced == False) and (st.session_state.fit == False) and (st.session_state.preprocessing_done == True) and (st.session_state.is_ready_for_fit == False) ):

                    st.header('Confirm Data Types')
                    transition_df = st.session_state.datatypes_df.copy().T
                    # Restore detection results from session state on reruns
                    new_len = len(st.session_state.data)
                    date_columns = st.session_state.date_columns
                    native_date_cols = st.session_state.native_date_cols
                    over_unique_threshold = st.session_state.over_unique_threshold

                    # identify  and filter observations that are the same in both columns
                    orig_str = transition_df['original_dtype'].astype(str)
                    poss_str = transition_df['possible_change'].astype(str)
                    mask_1 = (orig_str != poss_str)
                    mask_2 = ~(orig_str.isin(['category','object']) & poss_str.isin(['category','object']))
                    transition_df = transition_df.loc[mask_1 & mask_2]
                    st.markdown('Check Boxes to Approve Date Type Changes')
                    st.session_state.user_choice_datatype_changes = st.data_editor(transition_df)

                    if st.session_state.fit==True:
                        st.markdown('Changes Happen Prior to Fit. New Changes Require Re-Fitting of the Data')
                    make_changes = st.button('Make Changes')

                    if make_changes:
                        change_map = st.session_state.user_choice_datatype_changes[['possible_change','accept_change']].copy()
                        change_map = change_map.loc[change_map['accept_change']==True]
                        # update datatypes
                        for col in change_map.index:
                            datatype = change_map.loc[col, 'possible_change']
                            if str(datatype).startswith('datetime') or  str(datatype).startswith('timedelta'):
                                if col in native_date_cols:
                                    continue
                                else:
                                    st.session_state.data[col] = pd.to_datetime(st.session_state.data[col], errors='coerce')
                            else:
                                st.session_state.data[col] = st.session_state.data[col].astype(datatype)

                        st.session_state.types_confirmed = True


                if  (  (not st.session_state.early_stop) and (st.session_state.date_columns) and (st.session_state.drops_complete == True) and (st.session_state.types_confirmed == True) and (st.session_state.dates_pieced == False) and (st.session_state.fit == False) and (st.session_state.preprocessing_done == True) and (st.session_state.is_ready_for_fit == False) ):
                    st.header("Part Dates\nEg: 'Year', 'Quarter', 'Month', ...\nOr Drop Dates")
                    procede_w_date_col_left, drop_dates_col_right = st.columns([1,2])
                    with procede_w_date_col_left:
                        procede_w_date = st.button("Part Dates")
                    with drop_dates_col_right:
                        drop_dates = st.button("Drop Dates")

                    # Restore detection results from session state on reruns
                    new_len = len(st.session_state.data)
                    date_columns = st.session_state.date_columns
                    native_date_cols = st.session_state.native_date_cols
                    over_unique_threshold = st.session_state.over_unique_threshold

                    # Limit to st.session_state.max_date_cols_used = None date columns
                    cols_to_drop = []
                    if len(date_columns) > st.session_state.max_date_cols_used:
                        # Keep first 2, drop others
                        cols_to_drop = date_columns[st.session_state.max_date_cols_used:]
                        date_columns = date_columns[:st.session_state.max_date_cols_used]

                    # extract useful info from date/datetime columns
                    if procede_w_date:
                        # Create categorical variables from dates
                        for col in date_columns:
                            curr_cols = set(st.session_state.data.columns)
                            try:
                                new_title = f'{col}_year'
                                if new_title in curr_cols:
                                    new_title=new_title+'_i'
                                while new_title in curr_cols:
                                    new_title = new_title + "i"
                                st.session_state.data[new_title] = st.session_state.data[col].dt.year.astype('category')
                                if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                            except:
                                pass
                            try:
                                new_title = f'{col}_quarter'
                                if new_title in curr_cols:
                                    new_title=new_title+'_i'
                                while new_title in curr_cols:
                                    new_title = new_title + "i"
                                st.session_state.data[new_title] = st.session_state.data[col].dt.quarter.astype('category')
                                if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                            except:
                                pass
                            try:
                                new_title = f'{col}_month'
                                if new_title in curr_cols:
                                    new_title=new_title+'_i'
                                while new_title in curr_cols:
                                    new_title = new_title + "i"
                                st.session_state.data[new_title] = st.session_state.data[col].dt.month.astype('category')
                                if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                            except:
                                pass
                            try:
                                new_title = f'{col}_day'
                                if new_title in curr_cols:
                                    new_title=new_title+'_i'
                                while new_title in curr_cols:
                                    new_title = new_title + "i"
                                st.session_state.data[new_title] = st.session_state.data[col].dt.day.astype('category')
                                if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                            except:
                                pass
                            try:
                                new_title = f'{col}_hour'
                                if new_title in curr_cols:
                                    new_title=new_title+'_i'
                                while new_title in curr_cols:
                                    new_title = new_title + "i"
                                st.session_state.data[new_title] = st.session_state.data[col].dt.hour.astype('category')
                                if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                            except:
                                pass
                            try:
                                new_title = f'{col}_30min'
                                if new_title in curr_cols:
                                    new_title=new_title+'_i'
                                while new_title in curr_cols:
                                    new_title = new_title + "i"
                                st.session_state.data[new_title] = (st.session_state.data[col].dt.minute // 30).astype('category')
                                if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                            except:
                                pass
                                
                    if procede_w_date or drop_dates:        
                        # Drop original date columns (full date/datetime only, not extracted parts like year/month/day)
                        full_date_cols_to_drop = date_columns + cols_to_drop
                        if full_date_cols_to_drop:
                            st.session_state.data = st.session_state.data.drop(columns=full_date_cols_to_drop,errors='ignore')
                            if len(st.session_state.data.shape)<2:
                                st.session_state.data = st.session_state.data.to_frame()

                        st.session_state.dates_pieced = True

                if  ( (not st.session_state.early_stop) and  (st.session_state.drops_complete == True) and (st.session_state.types_confirmed == True) and ((st.session_state.dates_pieced == True) or (not st.session_state.date_columns)) and (st.session_state.fit == False) and (st.session_state.preprocessing_done == True) and (st.session_state.is_ready_for_fit == False) ):

                            
                            # Drop nulls part 2
                            # revise new_len   # this can drop up to n% of un-dropped columns due to % allowance of pd.NaT in coerce errors
                            # DROPNA NOT REQUIRED. MODULE WILL HANDLE NAN INTERNALLY
                            #st.session_state.data = st.session_state.data.dropna()
                            new_len = len(st.session_state.data)
                            dropped = st.session_state.original_len  - new_len 
                            dropped_pct = (dropped / st.session_state.original_len) * 100 if st.session_state.original_len > 0 else 0
                        
                            st.success(f"Data processed successfully!")
                            st.info(f"Dropped {dropped} observations ({dropped_pct:.1f}%) due to null values.")
                            st.info(f"Dropped {st.session_state.original_shape[1]-st.session_state.data.shape[1]} columns.")
                            st.write(f"Original dataset shape: {st.session_state.original_shape}\nFinal dataset shape: {st.session_state.data.shape}")

                            st.session_state.is_ready_for_fit = True

            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

  
    #=================================================================================================================================================================


    else:
        st.info("Please upload a CSV file to begin analysis.")

    if st.session_state.is_ready_for_fit and (not st.session_state.fit):




        fit = st.button("Fit the Data" if st.session_state.fit==False else "Re-Fit")

        if fit:
            st.session_state.AD = AnalyzeDataset(multivariate_params = st.session_state.multivariate_params,
                                                    kruskal_assumption_check_params=st.session_state.kruskal_assumption_check_params,
                                                    anova_assumption_check_params = st.session_state.anova_assumption_check_params,
                                                    chi2_assumption_check_params = st.session_state.chi2_assumption_check_params,
                                                    supercat_subcat_params = st.session_state.supercat_subcat_params,
                                                    multivariate_concatenation_delimiter = st.session_state.multivariate_concatenation_delimiter,
                                                    numnum_meth_alpha_above_instructions=st.session_state.numnum_meth_alpha_above_instructions,
                                                    numcat_meth_alpha_above_instructions=st.session_state.numcat_meth_alpha_above_instructions,
                                                    catcat_meth_alpha_above_instructions=st.session_state.catcat_meth_alpha_above_instructions,
                                                    good_of_fit_uniform_test_instrucions=st.session_state.good_of_fit_uniform_test_instrucions,
                                                    normal_test_instructions=st.session_state.normal_test_instructions
                                                    )
            st.session_state.AD.fit_full_dataset_analysis(st.session_state.data,  
                                numeric_columns=None,         # None for autodetect
                                categoric_columns=None,       # None for autodetect
                                numeric_target=None,          # None to compute all numeric variables as targets
                                categoric_target=None,        # None to compute all categoric variables as targets                          
                                fit_good_of_fit=True,         # instruct to test categoric variables for uniform distribution
                                fit_normal=True,
                                fit_multivariates=False,       # instruct to test multivariate significance
                                fit_supercat_subcats=True)    # test for super categories with subcategories that partition other variables
            
         
            st.session_state.fit = True


            result_df = st.session_state.AD.column_relationships_df(st.session_state.data.columns).reset_index(drop=False)
            st.session_state.result_df = result_df[['Target', 'Distribution', 'FeatureColum(s)', 'Test(s)']].rename(columns={'Test(s)':'Test:Assumptions_Met'}).set_index(['Target','Distribution'])



    if (st.session_state.result_df is not None) and (st.session_state.fit):
        # display a sample
        n_sample_rows = 3
        st.markdown(f'{n_sample_rows} sample rows of values')
        st.dataframe(st.session_state.data.sample(min(n_sample_rows, len(st.session_state.data))))
        st.markdown(f'Hypothosis and Correlation Results')
        st.dataframe(st.session_state.result_df)
        

elif st.session_state.page in ["Group Visualizations", "Target Visualizations"]:


    # Sidebar for n_wide controls
    with st.sidebar:

        # Page navigation #
        st.header("Navigation")
        st.session_state.page = st.radio("Navigation", 
                                         ["Data Upload & Processing", "Group Visualizations", "Target Visualizations"], 
                                         index=["Data Upload & Processing", "Group Visualizations", "Target Visualizations"].index(st.session_state.page),
                                         key = 'navigation_control',
                                         on_change = set_page)

        st.header("Plot Layout Controls")
        
        # Number of plot axes per row
        n_axes_options = [5, 6, 7]
        selected_n_axes = st.selectbox("Number of plot axes per row", n_axes_options, index=n_axes_options.index(st.session_state.n_wide['n_wide'][0]) if st.session_state.n_wide['n_wide'][0] in n_axes_options else 0)
        
        # Ideal max bars per row
        max_bars_options = [20, 30, 40, 50]
        selected_max_bars = st.selectbox("Ideal max bars per row", max_bars_options, index=max_bars_options.index(st.session_state.n_wide['n_wide'][1]) if st.session_state.n_wide['n_wide'][1] in max_bars_options else 1)
        
        # Row height
        height_options = [3, 4, 5, 6]
        selected_height = st.selectbox("Row height (inches)", height_options, index=height_options.index(st.session_state.n_wide['n_wide'][2]) if st.session_state.n_wide['n_wide'][2] in height_options else 2)
        
        # Update n_wide
        st.session_state.n_wide['n_wide'] = [selected_n_axes, selected_max_bars, selected_height]   

        # Update all relevant params
        st.session_state.num_univar_params.update(st.session_state.n_wide)
        st.session_state.cat_univar_params.update(st.session_state.n_wide)
        st.session_state.catcat_bivar_params.update(st.session_state.n_wide)
        # N/A st.session_state.numnum_bivar_params.update(st.session_state.n_wide)
        st.session_state.numcat_bivar_params.update(st.session_state.n_wide)
        
        st.header("Numerical Univariate Parameters")
        
        # force_significant_bin_edges
        #force_options = ["None", "True"]
        #current_force = "True" if st.session_state.num_univar_params.get('force_significant_bin_edges') else "None"
        #selected_force = st.selectbox("Force significant bin edges", force_options, index=force_options.index(current_force))
        st.session_state.num_univar_params['force_significant_bin_edges'] = True #if selected_force == "True" else None
        
        # minimize_significant_bins
        minimize_options = ["False", "True"]
        current_minimize = "True" if st.session_state.num_univar_params.get('minimize_significant_bins') else "False"
        selected_minimize = st.selectbox("Minimize significant bins", minimize_options, index=minimize_options.index(current_minimize))
        st.session_state.num_univar_params['minimize_significant_bins'] = True if selected_minimize == "True" else None
        
        # include_multivariate
        # multivariate_options = ["True", "False"]
        # current_multivariate = "True" if st.session_state.num_univar_params.get('include_multivariate', True) else "False"
        # selected_multivariate = st.radio("Include multivariate in bin computations", multivariate_options, index=multivariate_options.index(current_multivariate))
        # st.session_state.num_univar_params['include_multivariate'] = selected_multivariate == "True"

        st.header("Numeric-Categoric Bivariate Parameters")
        
        # plot_type for numcat_bivar_params
        plot_type_options = ["box", "boxen", "violin"]
        current_plot_type = st.session_state.numcat_bivar_params.get('plot_type', 'boxen')
        selected_plot_type = st.selectbox("Plot type for numeric-categoric plots", plot_type_options, index=plot_type_options.index(current_plot_type) if current_plot_type in plot_type_options else 1)
        st.session_state.numcat_bivar_params['plot_type'] = selected_plot_type
        
        st.header("Numeric-Numeric Bivariate Parameters")
        
        # plot_type for numnum_bivar_params
        numnum_plot_type_options = ["joint", "scatter"]
        current_numnum_plot_type = st.session_state.numnum_bivar_params.get('plot_type', 'joint')
        selected_numnum_plot_type = st.selectbox("Plot type for numeric-numeric plots", numnum_plot_type_options, index=numnum_plot_type_options.index(current_numnum_plot_type) if current_numnum_plot_type in numnum_plot_type_options else 0)
        st.session_state.numnum_bivar_params['plot_type'] = selected_numnum_plot_type
        
        # linreg for numnum_bivar_params
        linreg_options = ["False", "True"]
        current_linreg = "True" if st.session_state.numnum_bivar_params.get('linreg', False) else "False"
        selected_linreg = st.selectbox("Include linear regression line", linreg_options, index=linreg_options.index(current_linreg))
        st.session_state.numnum_bivar_params['linreg'] = selected_linreg == "True"
        
        st.header("Supercategory-Subcategory Parameters")
        
        # row_height for super_subcat_pairs_params
        row_height_options = [2, 3, 4, 5]
        current_row_height = st.session_state.super_subcat_pairs_params.get('row_height', 3)
        selected_row_height = st.selectbox("Row height (inches) for supersubcat plots", row_height_options, index=row_height_options.index(current_row_height) if current_row_height in row_height_options else 1)
        st.session_state.super_subcat_pairs_params['row_height'] = selected_row_height
        
        # cols_per_row for super_subcat_pairs_params
        cols_per_row_options = [1, 2, 3, 4]
        current_cols_per_row = st.session_state.super_subcat_pairs_params.get('cols_per_row', 2)
        selected_cols_per_row = st.selectbox("Columns per row for subcategory plots", cols_per_row_options, index=cols_per_row_options.index(current_cols_per_row) if current_cols_per_row in cols_per_row_options else 1)
        st.session_state.super_subcat_pairs_params['cols_per_row'] = selected_cols_per_row
        
        # y_tick_fontsize for super_subcat_pairs_params
        y_tick_fontsize_options = [8, 10, 12, 14, 16]
        current_y_tick_fontsize = st.session_state.super_subcat_pairs_params.get('y_tick_fontsize', 12)
        selected_y_tick_fontsize = st.selectbox("Y-axis tick font size", y_tick_fontsize_options, index=y_tick_fontsize_options.index(current_y_tick_fontsize) if current_y_tick_fontsize in y_tick_fontsize_options else 2)
        st.session_state.super_subcat_pairs_params['y_tick_fontsize'] = selected_y_tick_fontsize

  
    if not st.session_state.fit:
        st.info("The Data Hasn't Been Fit.")
    else:


        if st.session_state.page == "Group Visualizations":

            plot_overview_type_index_position = ['Numerical Non-Normal', 'Numerical Normal',
                        'Categorical Non-Uniform', 'Categorical Uniform',
                        'Numeric-Numeric With Correlation','Numeric-Numeric Without Correlation',
                        'Numeric-Categoric Reject Null', 'Numeric-Categoric Fail to Reject Null',
                        'Categoric-Categoric Reject Null','Categoric-Categoric Fail to Reject Null',
                        'Categoric Partitioned by Another Categoric',
                        'Assumptions Not Met - Categoric-Categoric',
                        'Assumptions Not Met - Numeric-Numeric',
                        'Assumptions Not Met - Numeric-Categoric',
                        'Assumptions Not Met - Numerical',
                        'Assumptions Not Met - Categorical'
                        ]
            try:
                plot_selection = st.segment_control("Select a Group to Plot", 
                                                plot_overview_type_index_position,
                                                selection_mode="single",
                                                default=None)
            except:
                plot_selection = st.selectbox("Select a Group to Plot", 
                                                plot_overview_type_index_position,
                                                index=None)
            if plot_selection:
                univar_and_bivar_col_lists = [
                                            {'num_univar':list(st.session_state.AD.reject_null_normal),
                                             'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Non-Normal'}} ,
                                            {'num_univar':list(st.session_state.AD.fail_to_reject_null_normal),
                                             'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Normal'}}  , 
                                            {'cat_univar':list(st.session_state.AD.reject_null_good_of_fit),
                                             'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Non-Uniform'}} ,  
                                            {'cat_univar':list(st.session_state.AD.fail_to_reject_null_good_of_fit),
                                             'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Uniform'}} , 
                                            {'numnum_bivar':st.session_state.AD.above_threshold_corr_numnum,
                                             'numnum_bivar_params':{**st.session_state.numnum_bivar_params, 'super_title':'Numeric-Numeric With Correlation'}}  , 
                                            {'numnum_bivar':st.session_state.AD.below_threshold_corr_numnum,
                                             'numnum_bivar_params':{**st.session_state.numnum_bivar_params,'super_title':'Numeric-Numeric Without Correlation'}} , 
                                            {'numcat_bivar':st.session_state.AD.reject_null_numcat,
                                             'numcat_bivar_params':{**st.session_state.numcat_bivar_params, 'super_title':'Numeric-Categoric Reject Null'}}  , 
                                            {'numcat_bivar': st.session_state.AD.fail_to_reject_null_numcat,
                                             'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Numeric-Categoric Fail to Reject Null'}}  , 
                                            {'catcat_bivar':st.session_state.AD.reject_null_catcat,
                                             'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Reject Null'}}  , 
                                            {'catcat_bivar':st.session_state.AD.fail_to_reject_null_catcat,
                                             'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Fail to Reject Null'}}  , 
                                            {'super_subcat_pairs':st.session_state.AD.supercategory_subcategory_pairs,
                                             'super_subcat_pairs_params':{**st.session_state.super_subcat_pairs_params,'super_title':'Categoric Partitioned by Another Categoric'}} ,                                             
                                             {'catcat_bivar':st.session_state.AD.assumptions_not_met['catcat'],
                                             'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Assumptions Not Met -- Categoric-Categoric'}},
                                             {'numnum_bivar':st.session_state.AD.assumptions_not_met['numnum'],
                                             'numnum_bivar_params':{**st.session_state.numnum_bivar_params,'super_title':'Assumptions Not Met -- Numeric-Numeric'}},
                                             {'numcat_bivar': st.session_state.AD.assumptions_not_met['numcat'],
                                             'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Assumptions Not Met -- Numeric-Categoric'}},
                                             {'num_univar':list(st.session_state.AD.assumptions_not_met['num']),
                                             'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Assumptions Not Met -- Numerical'}},
                                             {'cat_univar':list(st.session_state.AD.assumptions_not_met['cat']),
                                             'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Assumptions Not Met -- Categorical'}}
                                            ]

                default_params =  {
                                    'cat_univar':False,   
                                    'num_univar':False, 
                                    'catcat_bivar':False,
                                    'numnum_bivar':False,
                                    'numcat_bivar':False,
                                    'super_subcat_pairs':False,
                                    'cat_univar_params':st.session_state.cat_univar_params,
                                    'catcat_bivar_params': st.session_state.catcat_bivar_params,
                                    'numnum_bivar_params': st.session_state.numnum_bivar_params,
                                    'numcat_bivar_params':  st.session_state.numcat_bivar_params,
                                    'super_subcat_pairs_params': st.session_state.super_subcat_pairs_params,
                                    'num_univar_params': st.session_state.num_univar_params
                                    }        
                # update input parameters           
                default_params.update(univar_and_bivar_col_lists[plot_overview_type_index_position.index(plot_selection)])
                # call plot funciton
                st.session_state.AD.produce_all_plots(st.session_state.data,
                                            **default_params,
                                             streamlit_=True)
                

    
        elif st.session_state.page == "Target Visualizations":

            try:
                variable_selection = st.segment_control("Select Variables to Plot", 
                                        list(st.session_state.data.columns),
                                        selection_mode="multi",
                                        default=None)
            except:
                variable_selection = st.multiselect("Select Variables to Plot", 
                                        list(st.session_state.data.columns),
                                        default=None)

            

            available_ploting_options = [
                                        'Univariate - Numeric Non-Normal and Categorical Non-Uniform',
                                        'Numeric-Numeric With Correlation',
                                        'Numeric-Categoric Reject Null', 
                                        'Categoric-Categoric Reject Null',
                                        'Categoric Partitioned by Another Categoric'
                                        ]
            try:
                plot_type_selection = st.segment_control("Select a Plot Type", 
                                                        available_ploting_options,
                                                        selection_mode="multi")
            except:
                plot_type_selection = st.multiselect("Select a Plot Type", 
                                                        available_ploting_options,
                                                        default=None)

            plot_selections = st.button("Plot Selections")
            if plot_selections:
                if ( not variable_selection) or (not plot_type_selection):
                    st.info("No Selections to Plot")
                else:

                    param_col_lists = [
                                        {'not_uniform_or_reject_normal':True,
                                         'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Non-Normal'}
                                         ,'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Non-Uniform'}} ,   
                                        {'reject_numnum':True,'numnum_bivar_params':{**st.session_state.numnum_bivar_params,'super_title':'Numeric-Numeric With Correlation'}}  ,  
                                        {'reject_numcat':True,'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Numeric-Categoric Reject Null'}}  ,  
                                        {'reject_catcat':True,'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Reject Null'}}  ,  
                                        {'is_super_or_subcat':True,'super_subcat_pairs_params':{**st.session_state.super_subcat_pairs_params,'super_title':'Categoric Partitioned by Another Categoric'}}   
                                        ]
                    target_plot_default_params = {    
                                                    'reject_numcat': False,  
                                                    'reject_numnum': False,
                                                    'reject_catcat': False,
                                                    'is_super_or_subcat': False,
                                                    'not_uniform_or_reject_normal': False,  
                                                    'reject_multivariates': False,        
                                                    'auto_fit': True,   
                                                    'targets_share_plots': True ,  
                                                    'check_assumptions': True,
                                                    'dropna_gof': True,
                                                    'cat_univar_params':  st.session_state.cat_univar_params,
                                                    'catcat_bivar_params':  st.session_state.catcat_bivar_params,
                                                    'numnum_bivar_params':  st.session_state.numnum_bivar_params,
                                                    'numcat_bivar_params':  st.session_state.numcat_bivar_params,
                                                    'super_subcat_pairs_params':  st.session_state.super_subcat_pairs_params,
                                                    'num_univar_params':  st.session_state.num_univar_params
                                                    }
                    # use this to make sure values are present to plot
                    list_of_lists = [ list(st.session_state.AD.reject_null_normal)+list(st.session_state.AD.reject_null_good_of_fit),
                                      st.session_state.AD.above_threshold_corr_numnum,
                                      st.session_state.AD.reject_null_numcat,
                                      st.session_state.AD.reject_null_catcat,
                                      st.session_state.AD.supercategory_subcategory_pairs ] 
                    # only update parameters where variables are selected
                    index_list = []
                    for  i in range(len(available_ploting_options)):
                        if (available_ploting_options[i] in plot_type_selection):
                            if ( not list_of_lists[i] ):
                                st.warning(f"{available_ploting_options[i]} not present in the data")
                            else:
                                index_list.append(i)
                    for index in index_list:
                        target_plot_default_params.update(param_col_lists[index])

                    st.session_state.AD.visualize_by_targets(
                                            data=st.session_state.data,
                                            targets = list(variable_selection),
                                            **target_plot_default_params ,
                                             streamlit_=True )




