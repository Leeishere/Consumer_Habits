#file loader
from utils.BinnerClass import Bin
import pathlib
import pandas as pd
bin=Bin()
bin2=Bin()


def load_consumer_habits(filepath:str="utils/shopping_behavior_updated.csv"):
    """
    """
    global bin, bin2

    behavior=pathlib.Path(filepath)
    df=pd.read_csv(behavior)

    mymap={'Yes':1,'No':0}
    for col in ['Subscription Status', 'Discount Applied','Promo Code Used']:
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

    #bin numeric columns
    bin.relational_binner(df,max_cat_to_numeric_p=0.05,min_coeff=0.6,original_value_count_threashold=5,numeric_columns=None,categoric_columns=None)
    for k, v in bin.numeric_target_column_minimums.items():
        df[f"{k}_Binned"]=bin.binner(df[k],v)
        df[f"{k}_Binned"]=df[f"{k}_Binned"].astype('object')
    bin2.relational_binner(df,max_cat_to_numeric_p=0.05,min_coeff=0.6,original_value_count_threashold=5,numeric_columns=['Age'],categoric_columns=None)
    for k, v in bin2.numeric_target_column_minimums.items():
        df[f"{k}_Binned"]=bin2.binner(df[k],v)
        df[f"{k}_Binned"]=df[f"{k}_Binned"].astype('object')

    
    return df