import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# a probability function
def bayes_theorem(pa:pd.Series, pb_given_a:pd.Series, pb_given_not_a:pd.Series,use_log:bool=False):
    """
    input parameters P(A), P(B|A), P(B|NOT A), use_log:T/F
    returns P(A|B), where A is meant to be Review Rating level and B is meant to be a categoric variable
    """
    if use_log==False:
        numerator = pb_given_a * pa
        denominator = numerator + pb_given_not_a * (1 - pa)
        return numerator / denominator
    else:
        log_pa = np.log(pa)
        log_not_pa = np.log( 1-pa )
        log_num = np.log(pb_given_a) + log_pa
        log_denom = np.log(   np.exp(log_num) + np.exp(np.log(pb_given_not_a) +log_not_pa)  )
        return np.exp(log_num - log_denom)
# a function for plotting pareto charts
def plot_cat_to_review_levels(data,
                              category_column,
                              review_column,
                              probabilities:bool,
                              detail_plots:bool,
                              n_axis_columns,
                              use_log:bool,
                              dist_overview:bool):
    """
    where category column is a categorical columns such as 'Item Purchased'
    review_colunm is a review rating level such as Review Rating level 1, 2, or 3
    probabilities is boolean such as count/pareto charts vs probabilities
    detail_plots T/F determines if detail plots are wanted
    n_axis_columns is used if detail_plots is True. It is the number of plots to fit on one row when filling the figure
    use_log is passed to the bayes_theorem
    dist_overview True proceeds other plots with a distribution of each category
    """
    if dist_overview==True:
        plt.figure(figsize=(20,3.5))
        plt.title(f"Overview Distribution Of {category_column}")
        count_data=data[category_column].value_counts()
        sns.barplot(x=count_data.values,y=count_data.index,orient='h')
        plt.show()

    if probabilities==False and detail_plots==False and dist_overview==False:
        raise ValueError('one of (probabilities, detail_plots, dist_overview) needs to be True or there is nothing to return')
    # aggrigate and get probabilities
    grouped = data.groupby([category_column, review_column],as_index=False, observed=True).size().rename(columns={'size':'Count'})
    grouped[review_column] = grouped[review_column].astype(int)

    # get probabilites if True
    if probabilities==True:
        total_n = grouped['Count'].sum()
        review_totals = grouped.groupby(review_column)['Count'].transform('sum')
        category_totals = grouped.groupby(category_column)['Count'].transform('sum')
        # P(review)
        grouped['P_review'] = review_totals / total_n
        # P(category | review)
        grouped["P(Cat|Rev)"] = grouped['Count'] / review_totals
        # P(category | NOT review)
        grouped['P_category_given_not_review'] = (category_totals - grouped['Count']) / (total_n - review_totals) # where grouped['Count'] includes the rev meant to excude here
        grouped["P(Rev|Cat)"] = bayes_theorem(grouped['P_review'], grouped["P(Cat|Rev)"], grouped['P_category_given_not_review'],use_log=use_log)
        grouped.drop(columns=['P_category_given_not_review','Count','P_review'],inplace=True)
        del total_n, review_totals, category_totals
        plt.figure(figsize=(20,6))
        plt.title('Probability of Review Level Given Category')
        sns.barplot(data=grouped, x=category_column, y="P(Rev|Cat)", 
                    hue=review_column, hue_order=sorted(list(data[review_column].unique())), legend='auto')
    
    if detail_plots==True:
        num_axises = data[review_column].nunique()
        cols=n_axis_columns
        rows=int(np.ceil(num_axises/cols))
        fig = plt.figure(figsize=(20,(rows*3)+3))
        if probabilities==True:
            plt.suptitle(f"Probabilities of Review Level given Category and Category given Review Level\nWhere 1 is Lowest Review Rating Level\nDistributed Across {category_column}\n\n",fontsize=20)
        else:
            plt.suptitle(f"Pareto Plots Per Level of Review Rating\nWhere 1 is Lowest Rating Level and {data[review_column].max()} is Highest\nDistributed Across {category_column}\n\n",fontsize=20)
        for level in range(1,num_axises+1):
            if probabilities==True:
                plot_data=grouped.loc[grouped[review_column]==int(level)].sort_values(by="P(Cat|Rev)",ascending=False)
                plot_data = plot_data.melt( id_vars=[category_column], value_vars=[ "P(Rev|Cat)", "P(Cat|Rev)" ],var_name='Probability_Type', value_name='Probability' )
                ax = plt.subplot(rows,cols,level)
                ax.set_title(f"Review Rating Level {level}")
                legend='auto' if level in (1,3,5) else False
                sns.barplot( data=plot_data, x=category_column, y='Probability', hue='Probability_Type', ax=ax , legend=legend)
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylabel('Probability')
                ax.set_xlabel(f"P's accross lvl {level}")
            else:
                plot_data=grouped.loc[grouped[review_column]==int(level)].sort_values(by='Count',ascending=False)
                plot_data['Percent']=((plot_data['Count'].cumsum()) / plot_data['Count'].sum())*100
                ax = plt.subplot(rows,cols,level)
                ax.set_title(f"Review Rating Level {level}")
                x = np.arange(len(plot_data))

                ax.bar(x, plot_data['Count'])
                ax.set_xticks(x)
                ax.set_xticklabels(plot_data[category_column], rotation=45, ha='right')
                ax.set_ylabel('Count')
                ax.set_xlabel(f"Counts Per Category in Level {level}")
                ax2 = ax.twinx()
                ax2.plot(x, plot_data['Percent'], marker='o')
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('Cumulative Percent')

        plt.tight_layout()
        plt.show()