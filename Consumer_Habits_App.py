

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import stats,special
import streamlit as st

from utils.PoissonSalesForecasting import PoissonSalesForecasting
from utils.MuEstimator import MuEstimator
from utils.BinnerClass import Bin
from utils.CompareColumns import CompareColumns

import pathlib








#================================================================================================================================================================= 
# initial load and process data 
if 'filepath' not in st.session_state:
    st.session_state.filepath= pathlib.Path('utils') / 'shopping_behavior_updated.csv'
#@st.cache_data
def load_data(filepath=st.session_state.filepath):
    df=pd.read_csv(filepath)
    mymap={'Yes':1,'No':0}
    for col in ['Subscription Status', 'Discount Applied']:
        df[col]=df[col].map(mymap).astype('object')
    def freq_factor(x):
        if x=='Every 3 Months': return 365/4
        if x=='Annually': return 365/1
        if x=='Quarterly': return 365/4
        if x=='Monthly': return 365/12
        if x=='Bi-Weekly': return 7/2
        if x=='Fortnightly': return 14
        if x=='Weekly': return 7
    df['Total Days of Patronage']=(df['Frequency of Purchases'].map(freq_factor)*df['Previous Purchases']).astype(int)
    df=df.drop(columns='Customer ID')
    return df
#fetch data
if 'data' not in st.session_state:
    st.session_state.data = load_data()

# get headers
if 'unbinned_columns' not in st.session_state:
    st.session_state.unbinned_columns = None
if 'binned_columns' not in st.session_state:
    st.session_state.binned_columns = []
if 'binned_exists_as' not in st.session_state:
    st.session_state.binned_exists_as=None
if 'curr_original_unbinned' not in st.session_state:
    st.session_state.curr_original_unbinned = 'Review Rating'
if "numnum_metrics" not in st.session_state:
    st.session_state.numnum_metrics = ['pearson',0.6,True]
if "catnum_metrics" not in st.session_state:
    st.session_state.catnum_metrics = ['kruskal',0.05,False]
if "abs_min_bins" not in st.session_state:
    st.session_state.abs_min_bins = None
if "col_by_col_min_bins" not in st.session_state:
    st.session_state.col_by_col_min_bins = None
if "custom_min_bins" not in st.session_state:
    st.session_state.custom_min_bins = {}
if "choose_columns" not in st.session_state:
    st.session_state.choose_columns = False
if "binning_processed" not in st.session_state:
    st.session_state.binning_processed = False



#=================================================================================================================================================================

# PAGE CONFIG
st.set_page_config(
    page_title="Analyzing a Consumer Habits Dataset",
    page_icon="🛠️",
    layout="wide")

page = st.sidebar.radio("Navigate", ["Binning Tool", "Insights Gained w/ Binning Tool","Seasonal Forecasting Tool"])

#plt.style.use('seaborn-v0_8-colorblind')
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

if page == "Binning Tool":
    #=================================================================================================================================================================
    #TITLE
    st.title("Bin Variables Based on Hypothesis Tests and/or Correlation Coefficients.",text_alignment='center',)

    #=================================================================================================================================================================
    #BIN

    st.markdown('...',text_alignment='center')
    st.markdown("Minimum Bin Sizes that Retain Statistical Relationships",text_alignment='center')
    st.markdown('...',text_alignment='center')

    #=================================================================================================================================================================





    bin=Bin()

    def get_dict_for_min_bins(numeric_columns=None):
        global bin
        min_bin_dict=bin.relational_binner(st.session_state.data,
                                           numnum_meth_alpha_above=st.session_state.numnum_metrics,
                                           catnum_meth_alpha_above=st.session_state.catnum_metrics,
                                           original_value_count_threashold=5,
                                           numeric_columns=numeric_columns,
                                           categoric_columns=None,
                                           numeric_target=None,
                                           categoric_target=None )
        col_to_col_thresholds = bin.numeric_feature_col_thresholds
        return min_bin_dict, col_to_col_thresholds

    coco=CompareColumns()



    #@st.cache_data
    def update_data_with_binned_columns(data_,bin_dict):
        data=data_.copy()
        binned_list=[]
        for k, v in bin_dict.items():
            header=f"{k} -> Binned"
            data[header]=bin.binner(data[k],v,rescale=True,return_bins=False)
            data[header]=round(data[header],3).astype(float)    
            binned_list.append(header)
        return data,binned_list

    bin_selection_cell, col_to_col                          = st.columns([.5,.5],gap='large',vertical_alignment='top',border=True) 
    choose_custom_binsize, bin_the_vars                     = st.columns([.5,.5],gap='large',vertical_alignment='top',border=True) 
    examin_bins                                             =  st.columns([1],gap='large',vertical_alignment='top',border=True)
    pre_binned_lineplot_cell, pre_bin_relationships         = st.columns([.45,.55],gap='large',vertical_alignment='top',border=True)
    post_binned_countplot_cell,post_binned_relationships    = st.columns([.45,.55],gap='large',vertical_alignment='top',border=True)
    binned_mu_plot                                          = st.columns([1],gap='large',vertical_alignment='top',border=True) 

    # prefer stable, session-backed state over one-run flags

    # a cell that resets the bin metric dicts and selects a column to update/review
    with bin_selection_cell:
        st.markdown('Click to Bin Variables or to Reset Bins')
        reset = st.button("Reset Bins",width='stretch',key=2)
        if reset==True:
            # Reload original data without binned columns
            st.session_state.data = load_data()
            # Reset all binning-related session state
            st.session_state.abs_min_bins = None
            st.session_state.col_by_col_min_bins = None
            st.session_state.custom_min_bins = {}
            st.session_state.binned_columns = []
            st.session_state.binned_exists_as = None
            st.session_state.curr_original_unbinned = 'Review Rating'
            st.session_state.choose_columns = False
            st.session_state.binning_processed = False
            st.session_state.unbinned_columns = None
            # Now compute fresh bin minimums
            st.session_state.abs_min_bins, st.session_state.col_by_col_min_bins = get_dict_for_min_bins()
            st.session_state.custom_min_bins = st.session_state.abs_min_bins.copy()
            # Store the original numeric column names for selection
            st.session_state.unbinned_columns = list(st.session_state.custom_min_bins.keys())
            st.session_state.choose_columns = True
            st.rerun()
        # select box for columns in the custom bin dict
        col_in_review=None
        if st.session_state.choose_columns == True and st.session_state.unbinned_columns is not None:
            col_in_review = st.selectbox("Choose a Column to Bin.", st.session_state.unbinned_columns, index=None,key='examine_mins', 
                                        help=None, on_change=None, args=None, kwargs=None, placeholder=None, 
                                        disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")
    with col_to_col:   
        if col_in_review!=None: 
            st.subheader("Bin Sizes", divider='grey', anchor=False, text_alignment='center')
            st.markdown("Selected Variable's Binsizes Related to Others")
            threshold_df = pd.DataFrame(st.session_state.col_by_col_min_bins[col_in_review]).T.rename(columns={'min_within_threshold':'MinBins'})
            threshold_df = threshold_df['MinBins'].astype(int).to_frame()
            st.dataframe(data=threshold_df, width="stretch", height="auto",  use_container_width=None, hide_index=None, 
                            column_order=None, column_config=None, key=None, on_select="ignore", selection_mode="multi-row", 
                            row_height=None, placeholder=None)
    with choose_custom_binsize:   
        if col_in_review!=None:
            st.subheader("Pick a Bin Size", divider='grey', anchor=False, text_alignment='center')
            absMin=1
            new_binsize = st.number_input("Insert a number", step=1,min_value=absMin, value=max(absMin,5),key='get_custom_binsize')
            submit_new_binsize = st.button("Submit",width='stretch',key=222)
            if submit_new_binsize==True:
                st.session_state.custom_min_bins[col_in_review]=new_binsize
                st.info("Select More Variables or Continue.")
                submit_new_binsize=False
    with bin_the_vars:
        if st.session_state.abs_min_bins!=None:
            move_on_with_selected = st.button("Process",width='stretch',key='move_on_with_selected')
            st.markdown("Process Default & Custom Bins.")
            if move_on_with_selected==True:                 
                data=st.session_state.data.copy()
                st.session_state.data,binned_columns=update_data_with_binned_columns(data,st.session_state.custom_min_bins)
                del data
                # Can ensure that there where columns to bin. That's all st.session_state.binned_columns does.
                st.session_state.binned_columns+=binned_columns
                st.session_state.binning_processed = True
    
    examin_bins=examin_bins[0]
    with examin_bins:
        if st.session_state.binning_processed and (len(st.session_state.binned_columns)>0):
            st.subheader("Column Selection", divider='grey', anchor=False, text_alignment='center')
            st.markdown("Select the Binned Variable You Want to Examine.")
            try:
                unbinned_default_start_index=st.session_state.unbinned_columns.index(st.session_state.curr_original_unbinned)
            except:
                unbinned_default_start_index=0
            finally:
                unbinned = st.selectbox("", st.session_state.unbinned_columns, index=unbinned_default_start_index,key="selected_unbinned", 
                                help=None, on_change=None, args=None, kwargs=None, placeholder=None, 
                                disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")
                st.session_state.binned_exists_as = f"{unbinned} -> Binned"
                st.session_state.curr_original_unbinned = unbinned

    with pre_binned_lineplot_cell:
        if st.session_state.binned_exists_as!=None:
            st.subheader('Pre Bin Plot', divider='grey', anchor=False, text_alignment='center')
            st.markdown("Line Plot")
            unbinned_lineplot_data=st.session_state.data[st.session_state.curr_original_unbinned].copy().to_frame().reset_index(drop=True).reset_index(drop=False)#.sort_values(by=st.session_state.curr_original_unbinned,ascending=True)
            st.line_chart(data=unbinned_lineplot_data, x=unbinned_lineplot_data.columns[0], 
                          y=unbinned_lineplot_data.columns[1], x_label=None, y_label=st.session_state.curr_original_unbinned, color=None, 
                          width="stretch", height="content", use_container_width=None)
            

    with pre_bin_relationships: 
        if st.session_state.binned_exists_as!=None:
            st.subheader('Pre Bin Relationships', divider='grey', anchor=False, text_alignment='center')
            st.markdown("Metrics Before Binning")
            pre_combined=coco.column_comparison(st.session_state.data,
                            numnum_meth_alpha_above=st.session_state.numnum_metrics,
                            catnum_meth_alpha_above=st.session_state.catnum_metrics,
                            catcat_meth_alpha_above=None,
                            numeric_columns=None,
                            categoric_columns=None,
                            numeric_target=st.session_state.curr_original_unbinned,
                            categoric_target=None )
            pre_combined=pre_combined.loc[( (pre_combined['column_a']==st.session_state.curr_original_unbinned)|(pre_combined['column_b']==st.session_state.curr_original_unbinned) )&(pre_combined['column_a']!=pre_combined['column_b'])&~( (pre_combined['column_a']+" -> Binned"==pre_combined['column_b'])|(pre_combined['column_a']==pre_combined['column_b']+" -> Binned") )].round(3)
            pre_target_on_right = pre_combined['column_b'] == st.session_state.curr_original_unbinned
            pre_combined.loc[pre_target_on_right, ['column_a', 'column_b']] = ( pre_combined.loc[pre_target_on_right, ['column_b', 'column_a']].to_numpy())
            pre_combined=pre_combined.sort_values(by=['column_b'],ascending=[True]).drop_duplicates(subset=['column_a','column_b'],keep='first').reset_index(drop=True)
            st.dataframe(data=pre_combined[[col for col in pre_combined if col in ['column_b','P-value','Coefficient']]], width="stretch", height="auto",  use_container_width=None, hide_index=None, 
                         column_order=None, column_config=None, key=None, on_select="ignore", selection_mode="multi-row", 
                         row_height=None, placeholder=None) 
            

    with post_binned_countplot_cell:
        if st.session_state.binned_exists_as!=None:
            st.subheader('Post Bin Plot', divider='grey', anchor=False, text_alignment='center')
            st.markdown("Count Plot")
            binned_countplot_data=st.session_state.data[st.session_state.binned_exists_as].to_frame().groupby(st.session_state.binned_exists_as,as_index=False,observed=True).size().sort_values(by='size',ascending=False).reset_index(drop=True)
            st.bar_chart(data=binned_countplot_data, x=binned_countplot_data.columns[0], 
                         y=binned_countplot_data.columns[1], x_label=binned_countplot_data.columns[0], y_label="Counts", color=None, 
                         horizontal=False, sort=True, stack=None, width="stretch", height="content", use_container_width=None)
            


    with post_binned_relationships:
        if st.session_state.binned_exists_as!=None:
            st.subheader('Post Bin Relationships', divider='grey', anchor=False, text_alignment='center')
            st.markdown("Statistically Significant Relationships After Binning")
            post_combined=coco.column_comparison(st.session_state.data,
                            numnum_meth_alpha_above=st.session_state.numnum_metrics,
                            catnum_meth_alpha_above=st.session_state.catnum_metrics,
                            catcat_meth_alpha_above=None,
                            numeric_columns=None,
                            categoric_columns=None,
                            numeric_target=st.session_state.binned_exists_as,
                            categoric_target=None )
            post_combined=post_combined.loc[( (post_combined['column_a']==st.session_state.binned_exists_as)|(post_combined['column_b']==st.session_state.binned_exists_as) )&(post_combined['column_a']!=post_combined['column_b'])&~( (post_combined['column_a']+" -> Binned"==post_combined['column_b'])|(post_combined['column_a']==post_combined['column_b']+" -> Binned") )].round(3)
            target_on_right = post_combined['column_b'] == st.session_state.binned_exists_as
            post_combined.loc[target_on_right, ['column_a', 'column_b']] = ( post_combined.loc[target_on_right, ['column_b', 'column_a']].to_numpy())
            post_combined=post_combined.sort_values(by=['column_b'],ascending=[True]).drop_duplicates(subset=['column_a','column_b'],keep='first').reset_index(drop=True)
            st.dataframe(data=post_combined[[col for col in post_combined if col in ['column_b','P-value','Coefficient']]], width="stretch", height="auto", use_container_width=None, hide_index=None, 
                         column_order=None, column_config=None, key=None, on_select="ignore", selection_mode="multi-row", 
                         row_height=None, placeholder=None)  
            

    binned_mu_plot=binned_mu_plot[0]
    with binned_mu_plot:
        if st.session_state.binned_exists_as!=None:
            st.markdown("Bin Means")
            muest=MuEstimator()
            muest.get_floating_mean_hbar(st.session_state.data,st.session_state.curr_original_unbinned,0.95,[st.session_state.binned_exists_as],plot_title=None,median=False,streamlit=True)


    #=================================================================================================================================================================

    #mu estimator


    #=================================================================================================================================================================



    r_mu_and_relation_plot       = st.columns([1], gap="small", vertical_alignment="bottom", border=True, width="stretch")

    r_mu_and_relation_plot=r_mu_and_relation_plot[0]
    with r_mu_and_relation_plot:
        if st.session_state.binned_exists_as!=None:
            mu_column, partition_column =  st.session_state.curr_original_unbinned+" -> Binned", None
            muest2=MuEstimator()
            muest2.get_floating_proportion_hbar(st.session_state.data,mu_column,0.95,partition_column,plot_title=None,streamlit=True, proportion_within_partition=True)

elif page == "Seasonal Forecasting Tool":
    #=================================================================================================================================================================
    # FORECASTING
    st.markdown('...',text_alignment='center')
    st.header("Seasonal Forecasting Tool",text_alignment='center')
    st.markdown("Forecast Sales Based on the Poisson Probability Distribution.",text_alignment='center')
    st.markdown('...',text_alignment='center')

    #=================================================================================================================================================================

    model_psn=PoissonSalesForecasting()

    def pie_plot(df=st.session_state.data,period='month'):
        global model_psn
        model_psn.plot_mean_seasonal_sales(df,period,occurrence_multiplier=1.0,  total_prev_purchases = None, freq_of_purchases= None, 
                                           season_header_to_partition_sales_amounts = None, individual_sale_amounts = None,
                                           figure_figsize=(3,3), streamlit=True) # where figure_figsize:tuple|None
    def floating_cdf_plot(df=st.session_state.data,period='month'):
        global model_psn
        model_psn.floating_bar_plot(df,period,occurrence_multiplier=1.0,y_tick_aggregate_3rd_highest_nplace=0.5,  
                                    total_prev_purchases= None, freq_of_purchases = None, season_header_to_partition_sales_amounts= None,
                                    individual_sale_amounts= None, figure_figsize=(20,8), streamlit=True, auto_detect_height=True)

    col1a, col2a = st.columns([.25, .75],gap='large',vertical_alignment='top',border=True)
    col3a        = st.columns(1      ,gap='large',vertical_alignment='top',border=True)

    with col1a:
        st.subheader("Select", divider='grey', anchor=False, text_alignment='center')
        st.markdown("Pick a Period to Forecast Seasonal Sales By.")
        period_button = st.radio(
            "",
            ["week", "month", "season"],
            index=None,
            key="period_button",
            horizontal=False,
            label_visibility="collapsed",
        )

    with col2a:
        st.subheader("Forecasted Sales Per Period", divider='grey', anchor=False, text_alignment='center')
        if period_button is None:
            st.markdown('Please Select a Period to Forecast.')
        elif period_button is not None:
            pie_plot(df=st.session_state.data,period=period_button)# with button as input

    col3a=col3a[0]
    with col3a:
        st.subheader("A Full Year's Accumulative Sales Forecast", divider='grey', anchor=False, text_alignment='center')
        with st.container():
            if period_button is None:
                st.markdown('')
            elif period_button is not None:
                floating_cdf_plot(df=st.session_state.data,period=period_button)# with button as input

elif page == "Insights Gained w/ Binning Tool":


    Title_text = "Probability of Each Review Level Given a Categorical Variable."
    st.markdown('...',text_alignment='center')
    st.header(Title_text,text_alignment='center')
    st.markdown("Variables That Reject the Null Hypothesis Under the Kruskal-Wallis Test for Differneces Across Groups.",text_alignment='center')
    st.markdown('...',text_alignment='center')


    seven_levels_           = st.columns([1],gap='large',vertical_alignment='top',border=True) 
    Purchase_Amount_        = st.columns([1],gap='large',vertical_alignment='top',border=True) 
    color_                  =  st.columns([1],gap='large',vertical_alignment='top',border=True)
    ship_                   = st.columns([1],gap='large',vertical_alignment='top',border=True)
    sixteen_levels_         = st.columns([1],gap='large',vertical_alignment='top',border=True)
    size_                   = st.columns([1],gap='large',vertical_alignment='top',border=True) 

    with seven_levels_[0]:
        st.subheader("Review Ratings Binned Into 7 Levels", divider='grey', anchor=False, text_alignment='center')
        seven_levels_path = pathlib.Path("utils/imgs_Consumer_Habits_App/RevRat7Levels.png")
        st.image(seven_levels_path, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        
    with Purchase_Amount_[0]:
        st.subheader("Purchase Amount Binned and Ordinalized", divider='grey', anchor=False, text_alignment='center')
        Overview_Purch_Ords = pathlib.Path("utils/imgs_Consumer_Habits_App/Overview_Purch_Ords.png")
        st.image(Overview_Purch_Ords, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        st.markdown("Purchase Amount shows spikes in levels 1-3 for all purchase categories, but only spikes in high reviews for amounts 1, 3, and 4.")
        P_L_given_Purch_Ord = pathlib.Path("utils/imgs_Consumer_Habits_App/P_L_given_Purch_Ord.png")
        st.image(P_L_given_Purch_Ord, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        #st.markdown("Explore Low-Level Details.")    
        #Paretos_Purch_Ord = pathlib.Path("utils/imgs_Consumer_Habits_App/Paretos_Purch_Ord.png")
        #st.image(Paretos_Purch_Ord, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        
    with color_[0]:
        st.subheader("Color", divider='grey', anchor=False, text_alignment='center')
        Overview_Color = pathlib.Path("utils/imgs_Consumer_Habits_App/Overview_Color.png")
        st.image(Overview_Color, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        P_L_given_Color = pathlib.Path("utils/imgs_Consumer_Habits_App/P_L_given_Color.png")
        st.markdown("Many colors have high bias review levels.")
        st.image(P_L_given_Color, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        #Paretos_Color = pathlib.Path("utils/imgs_Consumer_Habits_App/Paretos_Color.png")
        #st.image(Paretos_Color, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
    
    with ship_[0]:
        st.subheader("Shipping Type", divider='grey', anchor=False, text_alignment='center')       
        Overview_Ship = pathlib.Path("utils/imgs_Consumer_Habits_App/Overview_Ship.png")
        st.image(Overview_Ship, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)   
        P_L_given_Ship = pathlib.Path("utils/imgs_Consumer_Habits_App/P_L_given_Ship.png")
        st.markdown("Standard shipping spikes for high reviews, but most other Shipping Types spike for low reviews.")
        st.image(P_L_given_Ship, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        Probas_Ship = pathlib.Path("utils/imgs_Consumer_Habits_App/Probas_Ship.png")
        st.markdown("For Shipping Types, probabilities for low Reviews tend to be high in both directions together.")
        st.image(Probas_Ship, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
    
    with sixteen_levels_[0]:
        st.subheader("Review Ratings Binned Into 16 Levels", divider='grey', anchor=False, text_alignment='center')
        RevRat16Levels = pathlib.Path("utils/imgs_Consumer_Habits_App/RevRat16Levels.png")
        st.image(RevRat16Levels, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
    
    with size_[0]:
        st.subheader("Size", divider='grey', anchor=False, text_alignment='center')
        Overview_Size = pathlib.Path("utils/imgs_Consumer_Habits_App/Overview_Size.png")
        st.image(Overview_Size, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        P_L_given_Size = pathlib.Path("utils/imgs_Consumer_Habits_App/P_L_given_Size.png")
        st.markdown("There is a slight tendency for L and M sizes to spike in low review levels and XL to spike in high. Spikes in S are not especially biased.")
        st.image(P_L_given_Size, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
        #Paretos_Size = pathlib.Path("utils/imgs_Consumer_Habits_App/Paretos_Size.png")
        #st.image(Paretos_Size, caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto",  use_container_width=None)
    
    
