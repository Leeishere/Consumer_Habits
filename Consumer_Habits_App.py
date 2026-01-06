

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
from utils.Chi2 import Chi2
from utils.ANOVA import ANOVA
from utils.Coefficient import Coefficient

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
if 'curr_related_columns' not in st.session_state:
    st.session_state.curr_related_columns = []
if 'captured_bin' not in st.session_state:
    st.session_state.captured_bin=None  # <--- this should only be updated when st.session_state.curr_related_columns is updated as well
if 'binned_exists_as' not in st.session_state:
    st.session_state.binned_exists_as=None
if 'partition_default' not in st.session_state:
    st.session_state.partition_default="Purchase Amount (USD)->BiNnEd"
if 'curr_but_original_unbinned' not in st.session_state:
    st.session_state.curr_but_original_unbinned = 'Review Rating'
if 'compare_relation_primary_value' not in st.session_state:
    st.session_state.compare_relation_primary = None


#=================================================================================================================================================================

# PAGE CONFIG
st.set_page_config(
    page_title="Analyzing a Consumer Habits Dataset",
    page_icon="🐍",
    layout="wide")

#plt.style.use('seaborn-v0_8-colorblind')
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

#=================================================================================================================================================================
#TITLE
st.title("Bin Variables and Forecast Sales Based On Consumer Habits",text_alignment='center',)

#=================================================================================================================================================================
#BIN

st.markdown('...',text_alignment='center')
st.header("Bin Continuous Values",text_alignment='center')
st.markdown("Minimum Bin Sizes that Retain Statistical Relationships",text_alignment='center')
st.markdown('...',text_alignment='center')

#=================================================================================================================================================================





bin=Bin()

def get_dict_for_min_bins(numeric_columns=None):
    global bin
    min_bin_dict=bin.relational_binner(st.session_state.data,max_cat_to_numeric_p=0.05,min_coeff=0.6,original_value_count_threashold=5,numeric_columns=numeric_columns,categoric_columns=None)
    return min_bin_dict
anova=ANOVA()
coeff=Coefficient()
chi=Chi2()


#@st.cache_data
def update_data_with_binned_columns(data_,bin_dict):
    data=data_.copy()
    binned_list=[]
    for key,val in bin_dict.items():
        header=f"{key}->BiNnEd"
        data[header]=bin.binner(data[key],val)
        data[header]=data[header].astype('object')
        binned_list.append(header)
    return data,binned_list

bin_selection_cell                                     = st.columns([1],gap='large',vertical_alignment='top',border=True) 
pre_binned_lineplot_cell, pre_bin_relationships         = st.columns([.65,.35],gap='large',vertical_alignment='top',border=True)
post_binned_countplot_cell,post_binned_relationships   = st.columns([.65,.35],gap='large',vertical_alignment='top',border=True)
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
            st.session_state.data,binned_columns=update_data_with_binned_columns(data,bin_dict)
            del data
            st.session_state.binned_columns+=binned_columns
            st.session_state.cat_columns+=binned_columns
            missing_numerics=[col for col in st.session_state.unbinned_columns if col not in bin_dict.keys()]
            if len(missing_numerics)>0:
                new_bins_dict=get_dict_for_min_bins(missing_numerics)
                bin_dict.update(new_bins_dict)
                data = st.session_state.data.copy()
                st.session_state.data,binned_columns=update_data_with_binned_columns(data,new_bins_dict) 
                del data
                st.session_state.binned_columns+=binned_columns
                st.session_state.cat_columns+=binned_columns
            if (len(st.session_state.binned_columns)>0):
                st.markdown("Select the Binned Variable You Want to Examine.")
                try:
                    unbinned_default_start_index=st.session_state.unbinned_columns.index(st.session_state.curr_but_original_unbinned)
                except:
                    unbinned_default_start_index=0
                finally:
                    unbinned = st.selectbox("", st.session_state.unbinned_columns, index=unbinned_default_start_index,key=1, 
                                    help=None, on_change=None, args=None, kwargs=None, placeholder=None, 
                                    disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")
                    st.session_state.binned_exists_as = f"{unbinned}->BiNnEd"
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
            st.session_state.binned_exists_as = f"{unbinned}->BiNnEd"
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
    st.markdown("Statistically Significant Relationships Before Binning")
    if b_pre_binned_lineplot_cell==False:
        st.info("",icon="🐍")
    else:
        pre_numcat=anova.cat_num_column_kruskal_wallis_relationships(st.session_state.data, alpha=0.05,reject=True, numeric_columns=[st.session_state.curr_but_original_unbinned],categoric_columns=None,detect_pseudo_numeric=True)
        pre_numcat.columns=['col_a','col_b','P-value']
        pre_numnum=coeff.num_num_column_spearman_coefficient_relationships(st.session_state.data, corr=0.6,reject=False,self_detect=True,numeric_columns=None,pseudo_numeric_columns=None,detect_pseudo_numeric=True)  
        pre_numnum.columns=['col_a','col_b','Coefficient']
        pre_combined=pd.concat([pre_numcat,pre_numnum])
        pre_combined=pre_combined.loc[( (pre_combined['col_a']==st.session_state.curr_but_original_unbinned)|(pre_combined['col_b']==st.session_state.curr_but_original_unbinned) )&(pre_combined['col_a']!=pre_combined['col_b'])&~( (pre_combined['col_a']+"->BiNnEd"==pre_combined['col_b'])|(pre_combined['col_a']==pre_combined['col_b']+"->BiNnEd") )].round(3)
        pre_target_on_right = pre_combined['col_b'] == st.session_state.curr_but_original_unbinned
        pre_combined.loc[pre_target_on_right, ['col_a', 'col_b']] = ( pre_combined.loc[pre_target_on_right, ['col_b', 'col_a']].to_numpy())
        pre_combined=pre_combined.sort_values(by=['col_b','P-value','Coefficient'],ascending=[True,False,True]).drop_duplicates(subset=['col_a','col_b'],keep='first').reset_index(drop=True)
        st.dataframe(data=pre_combined[['col_b','P-value','Coefficient']], width="stretch", height="auto",  use_container_width=None, hide_index=None, 
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
        post_numcat=anova.cat_num_column_kruskal_wallis_relationships(st.session_state.data, alpha=0.05,reject=True, numeric_columns=None,categoric_columns=None,detect_pseudo_numeric=True)
        post_numcat.columns=['col_a','col_b','P-value']
        post_numnum=coeff.num_num_column_spearman_coefficient_relationships(st.session_state.data, corr=0.6,reject=False,self_detect=True,numeric_columns=None,pseudo_numeric_columns=[st.session_state.binned_exists_as],detect_pseudo_numeric=True)  
        post_numnum.columns=['col_a','col_b','Coefficient']
        post_catcat=chi.categorical_column_relationships(st.session_state.data, alpha=0.05, columns=None, additional=True)
        post_catcat.columns=['col_a','col_b','P-value']
        post_catnum=anova.cat_num_column_kruskal_wallis_relationships(st.session_state.data, alpha=0.05,reject=True, numeric_columns=None,categoric_columns=[st.session_state.binned_exists_as],detect_pseudo_numeric=True)   
        post_catnum.columns=['col_a','col_b','P-value']
        post_combined=pd.concat([post_catcat,post_catnum,post_numcat,post_numnum])
        post_combined=post_combined.loc[( (post_combined['col_a']==st.session_state.binned_exists_as)|(post_combined['col_b']==st.session_state.binned_exists_as) )&(post_combined['col_a']!=post_combined['col_b'])&~( (post_combined['col_a']+"->BiNnEd"==post_combined['col_b'])|(post_combined['col_a']==post_combined['col_b']+"->BiNnEd") )].round(3)
        target_on_right = post_combined['col_b'] == st.session_state.binned_exists_as
        post_combined.loc[target_on_right, ['col_a', 'col_b']] = ( post_combined.loc[target_on_right, ['col_b', 'col_a']].to_numpy())
        post_combined=post_combined.sort_values(by=['col_b','P-value','Coefficient'],ascending=[True,False,True]).drop_duplicates(subset=['col_a','col_b'],keep='first').reset_index(drop=True)
        st.dataframe(data=post_combined[['col_b','P-value','Coefficient']], width="stretch", height="auto", use_container_width=None, hide_index=None, 
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
st.markdown('...',text_alignment='center')
st.header("Columnwise Relationships",text_alignment='center')
st.markdown("Examine Statistical Dependencies that Involve the New Binned Variable(s).",text_alignment='center')
st.markdown('...',text_alignment='center')

#=================================================================================================================================================================


r_select_mu_col, r_select_partition_col = st.columns([.5,.5], gap="small", vertical_alignment="bottom", border=True, width="stretch")
r_mu_and_relation_plot       = st.columns([1], gap="small", vertical_alignment="bottom", border=True, width="stretch")


rr_select_mu_col = False
rr_select_partition_col = False
rr_mu_and_relation_plot = False

with r_select_mu_col:
    # a condition check that indicates if binned in in the dataset
    st.subheader("Pick a Column to Examine it's Mean or Proportions.", divider='grey', anchor=False, text_alignment='center')    
    ### a bool check that directs the app to, or not to examine only current relationships as displayed above
    include_all=st.toggle("Examine Other Binned Variables", value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible", width="content")      
    if include_all==False:
        if b_binned_mu_plot==False:
            st.info("",icon="🐍")
        else:
            if (len(st.session_state.curr_related_columns)<1) or (st.session_state.binned_exists_as!=st.session_state.captured_bin):
                cols = ['col_a','col_b']
                master_relation_df=pd.concat(   [df[cols] for df in (post_combined, pre_combined)]  ) 
                st.session_state.curr_related_columns=list( set( list(master_relation_df['col_a'].unique())+list(master_relation_df['col_b'].unique()) ))
                st.session_state.captured_bin=st.session_state.binned_exists_as
          
            try:
                if st.session_state.captured_bin in st.session_state.curr_related_columns:
                    binned_proportion_default_index=st.session_state.curr_related_columns.index(st.session_state.captured_bin)
                else:
                    binned_proportion_default_index=st.session_state.curr_related_columns.index(st.session_state.curr_but_original_unbinned)
            except:
                binned_proportion_default_index=None

            st.session_state.compare_relation_primary=st.selectbox('', st.session_state.curr_related_columns, index=binned_proportion_default_index, key=None, 
                            help=None, on_change=None, args=None, kwargs=None, placeholder=None, 
                            disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")


            ###determine what plot func and what partition columns to offer
            if st.session_state.compare_relation_primary in st.session_state.unbinned_columns:
                get_mean_estimations=True   ########## to determine what plot func to call
            partition_options=[i for i in st.session_state.curr_related_columns if i != st.session_state.compare_relation_primary and i not in st.session_state.unbinned_columns]     

            rr_select_mu_col = True     

    else:
        if len(st.session_state.binned_columns)<1:
            st.info('',icon="")
        else:
            primary_var_options=st.session_state.binned_columns + st.session_state.unbinned_columns
            st.session_state.compare_relation_primary=st.selectbox('select', primary_var_options, index=0, key=None, 
                            help=None, on_change=None, args=None, kwargs=None, placeholder=None, 
                            disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")
            if st.session_state.compare_relation_primary in st.session_state.unbinned_columns:
                get_mean_estimations=True   ########## to determine what plot func to call
                st.session_state.partition_default=st.session_state.compare_relation_primary+"->BiNnEd"
                partition_options=[st.session_state.partition_default]

                rr_select_mu_col = True
            else: 
                st.session_state.partition_default=None
                partition_options=[]

                rr_select_mu_col = True

                


with r_select_partition_col: 
    st.markdown("Optional: Select Partition Column(s)")
    if rr_select_mu_col==False:
        st.info("",icon="🐍")
    else:        
        if st.session_state.compare_relation_primary == None:
            pass#st.info("",icon="🐍")
        else:
            if include_all==True:
                multi=st.multiselect("Partition By: ", partition_options,default=st.session_state.partition_default)

                rr_select_partition_col=True

            else:
                try:
                    if st.session_state.partition_default not in st.session_state.curr_related_columns:
                        if st.session_state.curr_related_columns[0] != st.session_state.compare_relation_primary:
                            st.session_state.partition_default=st.session_state.curr_related_columns[0]
                        else:
                            st.session_state.partition_default=st.session_state.curr_related_columns[1]
                    multi=st.multiselect("Partition By: ", partition_options,default=st.session_state.partition_default) #where default can be a of [None, list, or single value]
                    st.session_state.partition_default=multi.copy()

                    rr_select_partition_col=True

                except:
                    st.session_state.partition_default=None
                    multi=st.multiselect("Partition By: ", partition_options,default=st.session_state.partition_default) #where default can be a of [None, list, or singel value]
                    st.session_state.partition_default=multi.copy()

                    rr_select_partition_col=True

  

r_mu_and_relation_plot=r_mu_and_relation_plot[0]
with r_mu_and_relation_plot:
    if (rr_select_partition_col==False):# and (st.session_state.binned_exists_as!='Review Rating') and (st.session_state.partition_default!="Purchase Amount (USD)->BiNnEd"):
        st.info("",icon="🐍")
    else:
        try:
            if get_mean_estimations==True:
                muest2=MuEstimator()
                muest2.get_floating_mean_hbar(st.session_state.data,st.session_state.compare_relation_primary,0.95,multi,plot_title=None,streamlit=True)

                rr_mu_and_relation_plot=True

            else:
                muest2=MuEstimator()
                muest2.get_floating_proportion_hbar(st.session_state.data,st.session_state.compare_relation_primary,0.95,multi,plot_title=None,streamlit=True)

                rr_mu_and_relation_plot=True

        except:
            muest2=MuEstimator()
            muest2.get_floating_proportion_hbar(st.session_state.data,st.session_state.compare_relation_primary,0.95,multi,plot_title=None,streamlit=True)

            rr_mu_and_relation_plot=True



#=================================================================================================================================================================
# FORECASTING
st.markdown('...',text_alignment='center')
st.header("Seasonal Forecasting",text_alignment='center')
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
