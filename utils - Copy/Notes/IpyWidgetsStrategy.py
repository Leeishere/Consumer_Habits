

# notes for future ipywidgets strategy
# that puts the funciton into a dictionary

"""

-ipywidgets for notebooks:
import ipywidgets
from inpywidgets import interact
@interact
def toss_decision(team=list(df['toss_winner'].value_counts().index)):
    x=df.loc[(df['toss_winner']==team)]
    return x['toss_decision'].value_counts()
_______in a dict such as for utils.call:
def toss_decision(team):
    x = df.loc[df['toss_winner'] == team]
    return x['toss_decision'].value_counts()
my_dict = {    'my_func': interact(toss_decision,team=list(df['toss_winner'].value_counts().index))}
my_dict['my_func']

"""