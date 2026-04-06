

import numpy as np
import pandas as pd
import streamlit as st

from utils.AnalyzeDataset import AnalyzeDataset





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

# =================================================================================================

if "page" not in st.session_state:
    st.session_state.page = "Data Upload & Processing"
if "is_ready_for_fit" not in st.session_state:
    st.session_state.is_ready_for_fit = False
if "is_fit" not in st.session_state:
    st.session_state.is_fit = False

page = st.session_state.page
is_ready_for_fit = st.session_state.is_ready_for_fit
is_fit = st.session_state.is_fit

if page == "Data Upload & Processing":
    

    # Sidebar for threshold adjustments
    with st.sidebar:
        st.header("Threshold Adjustments")
        
        # Correlation thresholds for numeric-numeric
        current_corr = st.session_state.numnum_meth_alpha_above_instructions[0][1]
        corr_options = [current_corr,0.7,0.8,0.9]
        corr_labels = [f"{current_corr}"] + [str(i) for i in corr_options[1:]]
        selected_corr = st.radio("Correlation Threshold (Numeric-Numeric)", corr_labels, index=0)
        if selected_corr != corr_labels[0]:
            new_val = corr_options[corr_labels.index(selected_corr)]
            for item in st.session_state.numnum_meth_alpha_above_instructions:
                item[1] = new_val
        
        
        # P-value threshold for numeric-categoric
        current_p_numcat = st.session_state.numcat_meth_alpha_above_instructions[0][1]
        p_numcat_options = [current_p_numcat, 0.025, 0.01]
        p_numcat_labels = [f"{current_p_numcat}"] + [str(i) for i in p_numcat_options[1:]]
        selected_p_numcat = st.radio("P-value Threshold (Numeric-Categoric)", p_numcat_labels, index=0)
        if selected_p_numcat != p_numcat_labels[0]:
            new_val = p_numcat_options[p_numcat_labels.index(selected_p_numcat)]
            for item in st.session_state.numcat_meth_alpha_above_instructions:
                item[1] = new_val
        
        # P-value threshold for categoric-categoric
        current_p_catcat = st.session_state.catcat_meth_alpha_above_instructions[0][1]
        p_catcat_options = [current_p_catcat, 0.025, 0.01]
        p_catcat_labels = [f"{current_p_catcat}"] + [str(i) for i in p_catcat_options[1:]]
        selected_p_catcat = st.radio("P-value Threshold (Categoric-Categoric)", p_catcat_labels, index=0)
        if selected_p_catcat != p_catcat_labels[0]:
            new_val = p_catcat_options[p_catcat_labels.index(selected_p_catcat)]
            st.session_state.catcat_meth_alpha_above_instructions[0][1] = new_val
        
        # P-value threshold for good-of-fit uniform
        current_p_gof = st.session_state.good_of_fit_uniform_test_instrucions[0]
        p_gof_options = [current_p_gof, 0.025, 0.01]
        p_gof_labels = [f"{current_p_gof}"] + [str(i) for i in p_gof_options[1:]]
        selected_p_gof = st.radio("P-value Threshold (Good-of-Fit Uniform)", p_gof_labels, index=0)
        if selected_p_gof != p_gof_labels[0]:
            new_val = p_gof_options[p_gof_labels.index(selected_p_gof)]
            st.session_state.good_of_fit_uniform_test_instrucions[0] = new_val
        
        # P-value threshold for normal test
        current_p_norm = st.session_state.normal_test_instructions[0]
        p_norm_options = [current_p_norm, 0.025, 0.01]
        p_norm_labels = [f"{current_p_norm}"] + [str(i) for i in p_norm_options[1:]]
        selected_p_norm = st.radio("P-value Threshold (Normal Test)", p_norm_labels, index=0)
        if selected_p_norm != p_norm_labels[0]:
            new_val = p_norm_options[p_norm_labels.index(selected_p_norm)]
            st.session_state.normal_test_instructions[0] = new_val

        # decide to isolate partitioning super-subcat pairs
        curr_isolate = st.session_state.supercat_subcat_params["isolate_super_subs"]
        isolate_options = [curr_isolate, not curr_isolate]
        selected_isolate = st.radio("Isolate Partitioning Super-Subcategory Counterparts",isolate_options,index=0)
        if selected_isolate != st.session_state.supercat_subcat_params['isolate_super_subs']:
            st.session_state.supercat_subcat_params.update({'isolate_super_subs':selected_isolate })

        
        # return pseudo{'full_pseudo':False}
        curr_ret_p = st.session_state.kruskal_assumption_check_params['return_pseudo']
        retp_options = [curr_ret_p, not curr_ret_p]
        selected_retp = st.radio("Test Similarity Instead When Strict Mean Assumptions Aren't Met\nIn Kruskal-Wallis Test",retp_options,index=0)
        if selected_retp != st.session_state.kruskal_assumption_check_params['return_pseudo']:
            st.session_state.kruskal_assumption_check_params.update({'return_pseudo':selected_retp })

        # return pseudo
        curr_ps = st.session_state.kruskal_assumption_check_params['full_pseudo']
        ps_options = [curr_ps, not curr_ps]
        selected_ps = st.radio("Test Distribution Similarity Instead of Mean\nIn Kruskal-Wallis Test",ps_options,index=0)
        if selected_ps != st.session_state.kruskal_assumption_check_params['full_pseudo']:
            st.session_state.kruskal_assumption_check_params.update({'full_pseudo':selected_ps})

        # Page navigation #
        page = st.radio("Navigate", ["Data Upload & Processing", "Model Fitting", "Group Visualizations", "Target Visualizations"], index=["Data Upload & Processing", "Model Fitting", "Group Visualizations", "Target Visualizations"].index(st.session_state.page))
        st.session_state.page = page



    #=================================================================================================================================================================

    # CSV File Upload and Processing
    st.title("Data Analyzer App")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="File size limit: 10MB")

    if uploaded_file is not None:
        # Check file size (10MB limit for fly.io)
        file_size = len(uploaded_file.getvalue())
        if file_size > 10 * 1024 * 1024:
            st.error("File size exceeds 10MB limit. Please upload a smaller file.")
        else:
            try:
                # Read CSV to DataFrame
                df = pd.read_csv(uploaded_file)
                
                # Robust datatype detection
                for col in df.columns:
                    # Try to detect dates/datetimes
                    if df[col].dtype == 'object':
                        try:
                            # Attempt to parse as datetime
                            pd.to_datetime(df[col], errors='coerce')
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                    
                    # Detect categoric (if few unique values relative to total)
                    if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.1:
                        df[col] = df[col].astype('category')
                    elif df[col].dtype in ['int64', 'float64']:
                        # Keep numeric as is
                        pass
                
                # Identify date/datetime columns
                date_cols = []
                
                # Limit to 2 date columns
                if len(date_cols) > 2:
                    # Keep first 2, drop others
                    cols_to_drop = date_cols[2:]
                    df = df.drop(columns=cols_to_drop)
                    date_cols = date_cols[:2]
                    st.warning(f"Dropped extra date columns: {cols_to_drop}")
                
                # Create categorical variables from dates
                for col in date_cols:
                    try:
                        df[f'{col}_year'] = df[col].dt.year.astype('category')
                    except:
                        pass
                    try:
                        df[f'{col}_quarter'] = df[col].dt.quarter.astype('category')
                    except:
                        pass
                    try:
                        df[f'{col}_month'] = df[col].dt.month.astype('category')
                    except:
                        pass
                    try:
                        df[f'{col}_day'] = df[col].dt.day.astype('category')
                    except:
                        pass
                    try:
                        df[f'{col}_hour'] = df[col].dt.hour.astype('category')
                    except:
                        pass
                    try:
                        df[f'{col}_30min'] = (df[col].dt.minute // 30).astype('category')
                    except:
                        pass
                    
                
                # Drop original date columns
                df = df.drop(columns=date_cols)
                
                # Drop nulls
                original_len = len(df)
                df = df.dropna()
                dropped = original_len - len(df)
                dropped_pct = (dropped / original_len) * 100 if original_len > 0 else 0
                
                st.success(f"Data processed successfully!")
                st.info(f"Dropped {dropped} observations ({dropped_pct:.1f}%) due to null values.")
                st.write(f"Final dataset shape: {df.shape}")
                
                # Store in session state
                st.session_state.data = df
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

        st.session_state.is_ready_for_fit = True
        is_ready_for_fit = True

    else:
        st.info("Please upload a CSV file to begin analysis.")


  
    #=================================================================================================================================================================


if page == "Model Fitting":
    if not is_ready_for_fit:
        st.info("Data not Accessible.")
    else:
        first_fit = False
        refit = False
        if is_fit:
            refit = st.button("Re-Fit the Data")
        else:
            first_fit = st.button("Fit the Data")
        if first_fit and not is_fit:
            if "AD" not in st.session_state:
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

        elif refit and not first_fit:

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
        if refit or first_fit:
            st.session_state.AD.fit_full_dataset_analysis(st.session_state.data,  
                                numeric_columns=None,         # None for autodetect
                                categoric_columns=None,       # None for autodetect
                                numeric_target=None,          # None to compute all numeric variables as targets
                                categoric_target=None,        # None to compute all categoric variables as targets                          
                                fit_good_of_fit=True,         # instruct to test categoric variables for uniform distribution
                                fit_normal=True,
                                fit_multivariates=False,       # instruct to test multivariate significance
                                fit_supercat_subcats=True)    # test for super categories with subcategories that partition other variables

            st.session_state.is_fit = True
            is_fit = True
            first_fit = False
            refit = False

  

if page in ["Group Visualizations", "Target Visualizations"]:
    if not is_fit:
        st.info("The Data Hasn't Been Fit.")
    else:

        if 'n_wide' not in st.session_state:
            st.session_state.n_wide = {'n_wide': [8, 30, 4]}

        # Sidebar for n_wide controls
        with st.sidebar:
            st.header("Plot Layout Controls")
            
            # Number of plot axes per row
            n_axes_options = [5, 6, 7]
            selected_n_axes = st.radio("Number of plot axes per row", n_axes_options, index=n_axes_options.index(st.session_state.n_wide['n_wide'][0]) if st.session_state.n_wide['n_wide'][0] in n_axes_options else 0)
            
            # Ideal max bars per row
            max_bars_options = [20, 30, 40, 50]
            selected_max_bars = st.radio("Ideal max bars per row", max_bars_options, index=max_bars_options.index(st.session_state.n_wide['n_wide'][1]) if st.session_state.n_wide['n_wide'][1] in max_bars_options else 1)
            
            # Row height
            height_options = [3, 4, 5, 6]
            selected_height = st.radio("Row height (inches)", height_options, index=height_options.index(st.session_state.n_wide['n_wide'][2]) if st.session_state.n_wide['n_wide'][2] in height_options else 2)
            
            # Update n_wide
            st.session_state.n_wide['n_wide'] = [selected_n_axes, selected_max_bars, selected_height]   

            # Update all relevant params
            st.session_state.num_univar_params.update(st.session_state.n_wide)
            st.session_state.cat_univar_params.update(st.session_state.n_wide)
            st.session_state.catcat_bivar_params.update(st.session_state.n_wide)
            st.session_state.numnum_bivar_params.update(st.session_state.n_wide)
            st.session_state.numcat_bivar_params.update(st.session_state.n_wide)
            
            st.header("Numerical Univariate Parameters")
            
            # force_significant_bin_edges
            force_options = ["None", "True"]
            current_force = "True" if st.session_state.num_univar_params.get('force_significant_bin_edges') else "None"
            selected_force = st.radio("Force significant bin edges", force_options, index=force_options.index(current_force))
            st.session_state.num_univar_params['force_significant_bin_edges'] = True if selected_force == "True" else None
            
            # minimize_significant_bins
            minimize_options = ["None", "True"]
            current_minimize = "True" if st.session_state.num_univar_params.get('minimize_significant_bins') else "None"
            selected_minimize = st.radio("Minimize significant bins", minimize_options, index=minimize_options.index(current_minimize))
            st.session_state.num_univar_params['minimize_significant_bins'] = True if selected_minimize == "True" else None
            
            # include_multivariate
            multivariate_options = ["True", "False"]
            current_multivariate = "True" if st.session_state.num_univar_params.get('include_multivariate', True) else "False"
            selected_multivariate = st.radio("Include multivariate in bin computations", multivariate_options, index=multivariate_options.index(current_multivariate))
            st.session_state.num_univar_params['include_multivariate'] = selected_multivariate == "True"
            
            st.header("Numeric-Categoric Bivariate Parameters")
            
            # plot_type for numcat_bivar_params
            plot_type_options = ["box", "boxen", "violin"]
            current_plot_type = st.session_state.numcat_bivar_params.get('plot_type', 'boxen')
            selected_plot_type = st.radio("Plot type for numeric-categoric plots", plot_type_options, index=plot_type_options.index(current_plot_type) if current_plot_type in plot_type_options else 1)
            st.session_state.numcat_bivar_params['plot_type'] = selected_plot_type
            
            st.header("Numeric-Numeric Bivariate Parameters")
            
            # plot_type for numnum_bivar_params
            numnum_plot_type_options = ["joint", "scatter"]
            current_numnum_plot_type = st.session_state.numnum_bivar_params.get('plot_type', 'joint')
            selected_numnum_plot_type = st.radio("Plot type for numeric-numeric plots", numnum_plot_type_options, index=numnum_plot_type_options.index(current_numnum_plot_type) if current_numnum_plot_type in numnum_plot_type_options else 0)
            st.session_state.numnum_bivar_params['plot_type'] = selected_numnum_plot_type
            
            # linreg for numnum_bivar_params
            linreg_options = ["False", "True"]
            current_linreg = "True" if st.session_state.numnum_bivar_params.get('linreg', False) else "False"
            selected_linreg = st.radio("Include linear regression line", linreg_options, index=linreg_options.index(current_linreg))
            st.session_state.numnum_bivar_params['linreg'] = selected_linreg == "True"
            
            st.header("Supercategory-Subcategory Parameters")
            
            # row_height for super_subcat_pairs_params
            row_height_options = [2, 3, 4, 5]
            current_row_height = st.session_state.super_subcat_pairs_params.get('row_height', 3)
            selected_row_height = st.radio("Row height (inches) for supercategory plots", row_height_options, index=row_height_options.index(current_row_height) if current_row_height in row_height_options else 1)
            st.session_state.super_subcat_pairs_params['row_height'] = selected_row_height
            
            # cols_per_row for super_subcat_pairs_params
            cols_per_row_options = [1, 2, 3, 4]
            current_cols_per_row = st.session_state.super_subcat_pairs_params.get('cols_per_row', 2)
            selected_cols_per_row = st.radio("Columns per row for subcategory plots", cols_per_row_options, index=cols_per_row_options.index(current_cols_per_row) if current_cols_per_row in cols_per_row_options else 1)
            st.session_state.super_subcat_pairs_params['cols_per_row'] = selected_cols_per_row
            
            # y_tick_fontsize for super_subcat_pairs_params
            y_tick_fontsize_options = [8, 10, 12, 14, 16]
            current_y_tick_fontsize = st.session_state.super_subcat_pairs_params.get('y_tick_fontsize', 12)
            selected_y_tick_fontsize = st.radio("Y-axis tick font size", y_tick_fontsize_options, index=y_tick_fontsize_options.index(current_y_tick_fontsize) if current_y_tick_fontsize in y_tick_fontsize_options else 2)
            st.session_state.super_subcat_pairs_params['y_tick_fontsize'] = selected_y_tick_fontsize

            # Page navigation #
            page = st.radio("Navigate", ["Data Upload & Processing", "Model Fitting", "Group Visualizations", "Target Visualizations"], index=["Data Upload & Processing", "Model Fitting", "Group Visualizations", "Target Visualizations"].index(st.session_state.page))
            st.session_state.page = page

        if page == "Group Visualizations":

            plot_overview_type_index_position = ['Numerical Non-Normal', 'Numerical Normal',
                        'Categorical Non-Uniform', 'Categorical Uniform',
                        'Numeric-Numeric With Correlation','Numeric-Numeric Without Correlation',
                        'Numeric-Categoric Reject Null', 'Numeric-Categoric Fail to Reject Null',
                        'Categoric-Categoric Reject Null','Categoric-Categoric Fail to Reject Null',
                        'Categoric Partitioned by Another Categoric']

            plot_selection = st.segment_control("Select a Group to Plot", 
                                                plot_overview_type_index_position,
                                                selection_mode="single")

            univar_and_bivar_col_lists = [
                                        {'num_univar':st.session_state.AD.reject_null_normal,'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Non-Normal'}} ,
                                        {'num_univar':st.session_state.AD.fail_to_reject_null_normal,'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Normal'}}  , 
                                        {'cat_univar':st.session_state.AD.reject_null_good_of_fit,'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Non-Uniform'}} ,  
                                        {'cat_univar':st.session_state.AD.fail_to_reject_null_good_of_fit,'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Uniform'}} , 
                                        {'numnum_bivar':st.session_state.AD.above_threshold_corr_numnum,'numnum_bivar_params':{**st.session_state_numnum_bivar_params,'super_title':'Numeric-Numeric With Correlation'}}  , 
                                        {'numnum_bivar':st.session_state.AD.below_threshold_corr_numnum,'numnum_bivar_params':{**st.session_state_numnum_bivar_params,'super_title':'Numeric-Numeric Without Correlation'}} , 
                                        {'numcat_bivar':st.session_state.AD.reject_null_numcat,'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Numeric-Categoric Reject Null'}}  , 
                                        {'numcat_bivar': st.session_state.AD.fail_to_reject_null_numcat,'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Numeric-Categoric Fail to Reject Null'}}  , 
                                        {'catcat_bivar':st.session_state.AD.reject_null_catcat,'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Reject Null'}}  , 
                                        {'catcat_bivar':st.session_state.AD.fail_to_reject_null_catcat,'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Fail to Reject Null'}}  , 
                                        {'super_subcat_pairs':st.session_state.AD.supercategory_subcategory_pairs,'super_subcat_pairs_params':{**st.session_state.super_subcat_pairs_params,'super_title':'Categoric Partitioned by Another Categoric'}}   
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
                                        **default_params)
            



        elif page == "Target Visualizations":

            variable_selection = st.segment_control("Select Variables to Plot", 
                                        list(st.session_state.data.columns),
                                        selection_mode="multi")
            

            available_ploting_options = [
                                        'Numerical Non-Normal', 
                                        'Categorical Non-Uniform',
                                        'Numeric-Numeric With Correlation',
                                        'Numeric-Categoric Reject Null', 
                                        'Categoric-Categoric Reject Null',
                                        'Categoric Partitioned by Another Categoric'
                                        ]
            plot_type_selection = st.segment_control("Select a Plot Type", 
                                                        available_ploting_options,
                                                        selection_mode="single")
            plot_selections = st.button("Plot Selections")
            if plot_selections:
                if ( not variable_selection) or (not plot_type_selection):
                    st.info("No Selections to Plot")
                else:

                    param_col_lists = [
                                        {'num_univar':st.session_state.AD.reject_null_normal,'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Non-Normal'}} , 
                                        {'cat_univar':st.session_state.AD.reject_null_good_of_fit,'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Non-Uniform'}} ,   
                                        {'numnum_bivar':st.session_state.AD.above_threshold_corr_numnum,'numnum_bivar_params':{**st.session_state_numnum_bivar_params,'super_title':'Numeric-Numeric With Correlation'}}  ,  
                                        {'numcat_bivar':st.session_state.AD.reject_null_numcat,'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Numeric-Categoric Reject Null'}}  ,  
                                        {'catcat_bivar':st.session_state.AD.reject_null_catcat,'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Reject Null'}}  ,  
                                        {'super_subcat_pairs':st.session_state.AD.supercategory_subcategory_pairs,'super_subcat_pairs_params':{**st.session_state.super_subcat_pairs_params,'super_title':'Categoric Partitioned by Another Categoric'}}   
                                        ]
                    target_plot_default_params = {    
                                                    'reject_numcat': True,  
                                                    'reject_numnum': True,
                                                    'reject_catcat': True,
                                                    'is_super_or_subcat': True,
                                                    'not_uniform_or_reject_normal': True,  
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
                    target_plot_default_params.update(param_col_lists[available_ploting_options.index(plot_type_selection)])

                    st.session_state.AD.visualize_by_targets(
                                            data=st.session_state.data,
                                            targets = list(variable_selection),
                                            **target_plot_default_params  )
            
                    


