
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def qqplot_manual(
    data,
    dist=stats.norm,
    dist_params=(),
    line='fit',
    ax=None,
    scatter_kws=None,
    line_kws=None
):
    """
    Create a manual Q–Q plot comparing sample data to a theoretical distribution.
    
    Parameters
    ----------
    data : array-like
        1D array of sample data values.
        
    dist : scipy.stats distribution, default=stats.norm
        Theoretical distribution to compare against. Must provide a .ppf method.
        Examples:
            stats.norm          # normal distribution
            stats.expon         # exponential
            stats.t(df=5)       # t-distribution
            stats.uniform       # uniform distribution
        
    dist_params : tuple, default=()
        Parameters passed to the distribution, e.g.:
            stats.t, dist_params=(5,)     # df=5
            stats.expon, dist_params=(0, 2)   # loc=0, scale=2
        
        These map directly to dist.ppf(p, *dist_params).
        
    line : {'fit', '45', None}, default='fit'
        Type of reference line:
            'fit' → least-squares regression line
            '45'  → line with slope=1, intercept=0
            None  → no line plotted
        
    ax : matplotlib Axes, optional
        Axes to plot into. If None, one is created.
    
    scatter_kws : dict, optional
        Keyword arguments passed to seaborn.scatterplot.
    
    line_kws : dict, optional
        Keyword arguments passed to matplotlib.plot for the reference line.
    
    Returns
    -------
    ax : matplotlib Axes
        The axes containing the QQ plot.
    
    Notes
    -----
    • The function manually computes quantiles rather than relying on SciPy/Statsmodels.
    • To use a non-normal distribution, simply pass a SciPy distribution and optional parameters.
        Example:
            qqplot_manual(data, dist=stats.expon, dist_params=(0, 2))
            qqplot_manual(data, dist=stats.t, dist_params=(5,))
    """

    data = np.asarray(data)
    n = len(data)
    if n == 0:
        raise ValueError("Data array is empty.")

    # 1. Sort sample quantiles
    sample_q = np.sort(data)

    # 2. Compute evenly spaced probabilities (exclude exact 0 and 1)
    p = (np.arange(1, n + 1) - 0.5) / n

    # 3. Theoretical quantiles using ppf
    theor_q = dist.ppf(p, *dist_params)

    # Plotting setup
    if ax is None:
        fig, ax = plt.subplots()

    if scatter_kws is None:
        scatter_kws = {'s': 20}
    if line_kws is None:
        line_kws = {'linewidth': 2}

    # 4. Scatter plot
    sns.scatterplot(x=theor_q, y=sample_q, ax=ax, **scatter_kws)

    # 5. Reference line
    if line == 'fit':
        m, b = np.polyfit(theor_q, sample_q, 1)
        ax.plot(theor_q, m*theor_q + b, **line_kws)
    elif line == '45':
        ax.plot(theor_q, theor_q, **line_kws)

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"Q–Q Plot vs {dist.name}")

    return ax


import scipy.stats



class DistributionAnalysis:
    def __init__(self):
        self.mean=None
        self.sigma=None
        self.length=None
        self.lower=None
        self.upper=None

    def fit(self,vector_):
        vector=np.array(vector_.copy())
        self.mean=vector.mean()
        self.sigma=vector.std()
        self.length=vector.shape[0] 
        self.lower=vector.min()
        self.upper=vector.max()

    # synthetic normal vector
    def normally_distributed_vector(self):
        return np.random.normal(loc=self.mean, scale=self.sigma, size=self.length)

    #synthetic uniform vector
    def uniformlly_distributed_vector(self):
        return np.random.uniform(low=self.lower, high=self.upper, size=self.length)

    # a vectorized application of uniform 




    #####################################################################
    # KS-TEST AND CHI-TEST FUNCTIONS THAT TEST AGAINST MANY DISTRIBUTIONS
    ####################################################################
    

    ## these need to be impemented without scipy.stats to ensure GPU exceleration

    # Kolmogorov–Smirnov Test (K-S Test)--> test against any distribution 
    #Emperical Fn(X)= 1/n*sum(1 if x_i>=x else 0 for all_x_i @ each_x_i)
    #kstest takes the max diff from cdf and emperical function and that is the D statistic
    #could be written vectorized as a vector with sorted values, reset_index(drop=True), proportion (ie count >= over n) as (index+1)/total, and the cdf of the test distribution
    #https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html

    def test_vector_shape_kstest(self,vector,distributions:list=['uniform','normal']):
        """
        When data is > 2000 this is better than shapiro on tests of normality
        Default support for gaussian(normal) and uniform distributions
        other distrubutions: ['expontial','logistic','weibull_min','lognorm','chi2','pareto',] #'beta','gamma',
        H0: Data may follow tested distribution
        returns a dictionary of p values with the input distrubution(s) used as keys
        """
        p_values={}
        if 'uniform' in distributions:
            _ , uniform_p_value = scipy.stats.kstest(vector ,scipy.stats.uniform(loc=self.lower,scale=self.upper-self.lower).cdf)  #  I want to plot this
            p_values['uniform']=uniform_p_value
        if 'normal' or 'gaussian' in distributions:
            _ , norm_p_value = scipy.stats.kstest(vector, scipy.stats.norm(loc=self.mean, scale=self.sigma).cdf)
            if 'uniform' in distributions:
                p_values['normal']=norm_p_value
            if 'gaussian' in distributions:
                p_values['gaussian']=norm_p_value
        return p_values      
    

    

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html#scipy.stats.shapiro
    #Shapiro-Wilk Test:  test normality on small to medium-sized datasets < 2000 samples. 
    def shapiro_normality_test(self,data):
        """
        H0: data is normally distributed.  
        """ 
        _ , p_value = scipy.stats.shapiro(data)
        return p_value
 

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
    def anderson_darling_for_normality(self,data):
        """
        datasets of all sizes,
        similar to ks-test, but gives more weight to tails
        """
        result = scipy.stats.anderson(data, dist='norm')
        # Display results
        print("Test Statistic:", result.statistic)
        print("Critical Values:", result.critical_values)
        print("Significance Levels:", result.significance_level)

#use column_makeup() and filtered_makeup() with these
# plot multiple brobablistic models from bellow onto observed
# return chi p_values and kstest p values  
# return skew and curtosis  
for col in ['Purchase Amount (USD)','Review Rating','Previous Purchases']:
    d_analyzer=DistributionAnalysis()
    vector=df[col].sort_values()
    d_analyzer.fit(vector)
    print(col)
    print(d_analyzer.test_vector_shape_kstest(df[col],distributions=['uniform','normal']))

    normalized=d_analyzer.normally_distributed_vector()
    uniform=d_analyzer.uniformlly_distributed_vector()
    fig, ax = plt.subplots()

    sns.histplot(vector,label='hist',ax=ax,stat='density', color='gray', bins=6, kde=False)
    sns.kdeplot(normalized,label='norm',ax=ax)
    sns.kdeplot(uniform,label='uniform',ax=ax)
    sns.kdeplot(vector,label='observed',color='black',ax=ax)
    plt.legend()
    plt.show()

# a boxeplot funciton that shows outlier removal haxis lines at given ntiles [.005,.01,.025, .05, .075, .1, .9, .975, .95, .975,.99,.995] or np.percentiles(vector, [ 1,2.5,  5. ,  7.5, 10. , 90. , 97.5, 95. , 97.5,99])
# it should lay them out in one wide column such that the var is y and val is x [or with tall fixed height]

# joint plot

# calulate skew and curtosistest normality within groups-->hist plots, lots of tests