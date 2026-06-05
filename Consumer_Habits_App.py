

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import stats,special
import streamlit as st

from data_analysis_utils.PoissonSalesForecasting import PoissonSalesForecasting

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

#=================================================================================================================================================================

# PAGE CONFIG
st.set_page_config(
    page_title="Analyzing a Consumer Habits Dataset",
    page_icon="🛠️",
    layout="wide")

page = st.sidebar.radio("Navigate", ["Seasonal Forecast"])

#plt.style.use('seaborn-v0_8-colorblind')
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

if page == "Seasonal Forecast":
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

    
