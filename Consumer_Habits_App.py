

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
if 'cat_columns' not in st.session_state:
    st.session_state.cat_columns = list(st.session_state.data.select_dtypes(include=['category', 'object']).columns)
if 'unbinned_columns' not in st.session_state:
    st.session_state.unbinned_columns = list(st.session_state.data.select_dtypes([int,float]).columns)
if 'binned_columns' not in st.session_state:
    st.session_state.binned_columns = []
if 'binned_exists_as' not in st.session_state:
    st.session_state.binned_exists_as=None
if 'curr_but_original_unbinned' not in st.session_state:
    st.session_state.curr_but_original_unbinned = 'Review Rating'
if 'min_num_bins_to_implement' not in st.session_state:
    st.session_state.min_num_bins_to_implement = 5
if "numnum_metrics" not in st.session_state:
    st.session_state.numnum_metrics = ['welch',0.05,False]
if "catnum_metrics" not in st.session_state:
    st.session_state.catnum_metrics = ['kruskal',0.05,False]


#=================================================================================================================================================================

# PAGE CONFIG
st.set_page_config(
    page_title="Analyzing a Consumer Habits Dataset",
    page_icon="🐍",
    layout="wide")

page = st.sidebar.radio("Navigate", ["Binning Tool", "Seasonal Forecasting Tool"])

#plt.style.use('seaborn-v0_8-colorblind')
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

if page == "Binning Tool":
    #=================================================================================================================================================================
    #TITLE
    st.title("A Binning Tool That Considers Hypothesis Tests and/or Correlation Coefficients.",text_alignment='center',)

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
        return min_bin_dict

    coco=CompareColumns()



    #@st.cache_data
    def update_data_with_binned_columns(data_,bin_dict,min_bins):
        data=data_.copy()
        binned_list=[]
        for k, v in bin_dict.items():
            v=max(v,min_bins)# set the minimun number of bins for any binned columns
            header=f"{k} -> Binned"
            data[header]=bin.binner(data[k],v,rescale=True,return_bins=False)
            data[header]=round(data[header],3).astype(float)    
            binned_list.append(header)
        return data,binned_list

    bin_selection_cell                                     = st.columns([1],gap='large',vertical_alignment='top',border=True) 
    pre_binned_lineplot_cell, pre_bin_relationships         = st.columns([.45,.55],gap='large',vertical_alignment='top',border=True)
    post_binned_countplot_cell,post_binned_relationships   = st.columns([.45,.55],gap='large',vertical_alignment='top',border=True)
    binned_mu_plot                                         = st.columns([1],gap='large',vertical_alignment='top',border=True) 


    b_bin_selection_cell = False
    b_pre_bin_relationships = False
    b_pre_binned_lineplot_cell = False
    b_post_binned_relationships = False
    b_post_binned_countplot_cell = False
    b_binned_mu_plot = False

    bin_selection_cell=bin_selection_cell[0]
    with bin_selection_cell:
        if len(st.session_state.binned_columns)<1:        
            st.subheader("Bin Continuous Variables", divider='grey', anchor=False, text_alignment='center')
            get_binned_columns=True   #st.button("Click Here",width='stretch',key=2)
            if get_binned_columns:
                bin_dict=get_dict_for_min_bins()
                data=st.session_state.data.copy()
                st.session_state.data,binned_columns=update_data_with_binned_columns(data,bin_dict,st.session_state.min_num_bins_to_implement)
                del data
                st.session_state.binned_columns+=binned_columns
                st.session_state.cat_columns+=binned_columns
                missing_numerics=[col for col in st.session_state.unbinned_columns if col not in bin_dict.keys()]
                if len(missing_numerics)>0:
                    new_bins_dict=get_dict_for_min_bins(missing_numerics)
                    bin_dict.update(new_bins_dict)
                    data = st.session_state.data.copy()
                    st.session_state.data,binned_columns=update_data_with_binned_columns(data,new_bins_dict,st.session_state.min_num_bins_to_implement) 
                    del data
                    st.session_state.binned_columns+=binned_columns
                    st.session_state.cat_columns+=binned_columns
                if (len(st.session_state.binned_columns)>0):
                    st.markdown("Select the Variable to Examine.")
                    try:
                        unbinned_default_start_index=st.session_state.unbinned_columns.index(st.session_state.curr_but_original_unbinned)
                    except:
                        unbinned_default_start_index=0
                    finally:
                        unbinned = st.selectbox("", st.session_state.unbinned_columns, index=unbinned_default_start_index,key=1, 
                                        help=None, on_change=None, args=None, kwargs=None, placeholder=None, 
                                        disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")
                        st.session_state.binned_exists_as = f"{unbinned} -> Binned"
                        st.session_state.curr_but_original_unbinned = unbinned  
                
                    b_bin_selection_cell = True

        else:  
            st.subheader("Column Selection", divider='grey', anchor=False, text_alignment='center')
            st.markdown("Select the Binned Variable You Want to Examine.")
            try:
                unbinned_default_start_index=st.session_state.unbinned_columns.index(st.session_state.curr_but_original_unbinned)
            except:
                unbinned_default_start_index=0
            finally:
                unbinned = st.selectbox("", st.session_state.unbinned_columns, index=unbinned_default_start_index,key=None, 
                                help=None, on_change=None, args=None, kwargs=None, placeholder=None, 
                                disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")
                st.session_state.binned_exists_as = f"{unbinned} -> Binned"
                st.session_state.curr_but_original_unbinned = unbinned
        
            b_bin_selection_cell = True

    with pre_binned_lineplot_cell:
        st.subheader('Pre Bin Plot', divider='grey', anchor=False, text_alignment='center')
        st.markdown("Line Plot")
        if st.session_state.binned_exists_as==None:
            st.info("",icon="🐍")
        else:
            unbinned_lineplot_data=st.session_state.data[st.session_state.curr_but_original_unbinned].copy().to_frame().reset_index(drop=True).reset_index(drop=False)#.sort_values(by=st.session_state.curr_but_original_unbinned,ascending=True)
            st.line_chart(data=unbinned_lineplot_data, x=unbinned_lineplot_data.columns[0], 
                          y=unbinned_lineplot_data.columns[1], x_label=None, y_label=st.session_state.curr_but_original_unbinned, color=None, 
                          width="stretch", height="content", use_container_width=None)
            
            b_pre_binned_lineplot_cell=True

    with pre_bin_relationships: 
        st.subheader('Pre Bin Relationships', divider='grey', anchor=False, text_alignment='center')
        st.markdown("Metrics Before Binning")
        if b_pre_binned_lineplot_cell==False:
            st.info("",icon="🐍")
        else:
            pre_combined=coco.column_comparison(st.session_state.data,
                            numnum_meth_alpha_above=st.session_state.numnum_metrics,
                            catnum_meth_alpha_above=st.session_state.catnum_metrics,
                            catcat_meth_alpha_above=None,
                            numeric_columns=None,
                            categoric_columns=None,
                            numeric_target=st.session_state.curr_but_original_unbinned,
                            categoric_target=None )
            pre_combined=pre_combined.loc[( (pre_combined['column_a']==st.session_state.curr_but_original_unbinned)|(pre_combined['column_b']==st.session_state.curr_but_original_unbinned) )&(pre_combined['column_a']!=pre_combined['column_b'])&~( (pre_combined['column_a']+" -> Binned"==pre_combined['column_b'])|(pre_combined['column_a']==pre_combined['column_b']+" -> Binned") )].round(3)
            pre_target_on_right = pre_combined['column_b'] == st.session_state.curr_but_original_unbinned
            pre_combined.loc[pre_target_on_right, ['column_a', 'column_b']] = ( pre_combined.loc[pre_target_on_right, ['column_b', 'column_a']].to_numpy())
            pre_combined=pre_combined.sort_values(by=['column_b'],ascending=[True]).drop_duplicates(subset=['column_a','column_b'],keep='first').reset_index(drop=True)
            st.dataframe(data=pre_combined[[col for col in pre_combined if col in ['column_b','P-value','Coefficient']]], width="stretch", height="auto",  use_container_width=None, hide_index=None, 
                         column_order=None, column_config=None, key=None, on_select="ignore", selection_mode="multi-row", 
                         row_height=None, placeholder=None) 
            
            b_pre_bin_relationships=True

    with post_binned_countplot_cell:
        st.subheader('Post Bin Plot', divider='grey', anchor=False, text_alignment='center')
        st.markdown("Count Plot")
        if b_pre_bin_relationships==False:
            st.info("",icon="🐍")
        else:
            binned_countplot_data=st.session_state.data[st.session_state.binned_exists_as].to_frame().groupby(st.session_state.binned_exists_as,as_index=False,observed=True).size().sort_values(by='size',ascending=False).reset_index(drop=True)
            st.bar_chart(data=binned_countplot_data, x=binned_countplot_data.columns[0], 
                         y=binned_countplot_data.columns[1], x_label=binned_countplot_data.columns[0], y_label="Counts", color=None, 
                         horizontal=False, sort=True, stack=None, width="stretch", height="content", use_container_width=None)
            
            b_post_binned_countplot_cell=True


    with post_binned_relationships:
        st.subheader('Post Bin Relationships', divider='grey', anchor=False, text_alignment='center')
        st.markdown("Statistically Significant Relationships After Binning")
        if b_post_binned_countplot_cell==False:
            st.info("",icon="🐍")
        else:
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
            
            b_post_binned_relationships=True          

    binned_mu_plot=binned_mu_plot[0]
    with binned_mu_plot:
        st.markdown("Bin Means")
        if b_post_binned_relationships==False:
            st.info("",icon="🐍")
        else:
            muest=MuEstimator()
            muest.get_floating_mean_hbar(st.session_state.data,st.session_state.curr_but_original_unbinned,0.95,[st.session_state.binned_exists_as],plot_title=None,median=False,streamlit=True)

            b_binned_mu_plot = True


    #=================================================================================================================================================================

    #mu estimator


    #=================================================================================================================================================================



    r_mu_and_relation_plot       = st.columns([1], gap="small", vertical_alignment="bottom", border=True, width="stretch")

    r_mu_and_relation_plot=r_mu_and_relation_plot[0]
    with r_mu_and_relation_plot:
        if (b_binned_mu_plot==False):
            st.info("",icon="🐍")
        else:
            mu_column, partition_column =  st.session_state.curr_but_original_unbinned+" -> Binned", None
            muest2=MuEstimator()
            muest2.get_floating_proportion_hbar(st.session_state.data,mu_column,0.95,partition_column,plot_title=None,streamlit=True, proportion_within_partition=True)

            rr_mu_and_relation_plot=True

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
        week_button=st.button("Forecast by Week",
                              type="primary",
                              use_container_width=True  )
        month_button=st.button("Forecast by Month",
                              type="primary",
                              use_container_width=True  )
        season_button=st.button("Forecast by Season",
                              type="primary",
                              use_container_width=True  )
        period_button=None
        if (week_button and month_button) or (week_button and season_button) or (month_button and season_button) or (week_button and month_button and season_button):
            st.info('Only One Period at a Time.\nPlease Deselect Button(s).', icon="🐍")
        elif (not week_button) and (not month_button) and (not season_button):
            if st.session_state.binned_exists_as==None:
                period_button=period_button  # no change
            else: period_button='month'
        elif week_button:
            period_button='week'
        elif month_button:
            period_button='month'
        elif season_button:
            period_button='season'

    with col2a:
        st.subheader("Forecasted Sales Per Period", divider='grey', anchor=False, text_alignment='center')
        if period_button is None:
            st.info('Please Select a Period to Forecast.', icon="🐍")
        elif period_button is not None:
            pie_plot(df=st.session_state.data,period=period_button)# with button as input

    col3a=col3a[0]
    with col3a:
        st.subheader("A Full Year's Accumulative Sales Forecast", divider='grey', anchor=False, text_alignment='center')
        with st.container():
            if period_button is None:
                st.info('', icon="🐍")
            elif period_button is not None:
                floating_cdf_plot(df=st.session_state.data,period=period_button)# with button as input
