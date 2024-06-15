#%% Importing Libraries and initial setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import warnings
import seaborn as sns
import tkinter as tk
from tkinter import simpledialog
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set(style="whitegrid", palette="husl")


#%% Download data from Yahoo Finance

def get_dates():
    root = tk.Tk()
    root.withdraw()  # Hides the main window

    start_date = simpledialog.askstring("Input", "Enter start date (YYYY-MM-DD):", parent=root)
    end_date = simpledialog.askstring("Input", "Enter end date (YYYY-MM-DD):", parent=root)
    
    root.destroy()  # Closes the Tkinter window

    return start_date, end_date


def import_commodity_data(tickers: list, start_date: str, end_date: str) -> dict:

    # Define commodity categories and their respective tickers
    commodity_dict = {
        
        "Energy": ["CL=F", "NG=F", "HO=F", "RB=F", 'BZ=F'],
        "Metals - Precious": ["GC=F", "SI=F", "PA=F", "PL=F"],
        "Metals - Base": ["HG=F", 'ZN=F'],  
        "Agriculture": ["ZW=F", "ZC=F", "ZS=F", "CC=F", 'ZR=F', 'ZO=F', 'GDK=F','DY=F', 'CB=F', 'GNF=F'],
        "Livestock": ["HE=F", "LE=F", "GF=F"],
        "Softs": ["SB=F", "KC=F", "CT=F"],
        "ETFs": ["DBB", 'USO', "DBA", 'DBC', 'GSG', 'RJI'] 

}

    # Create an empty dictionary to store data, and another for ticker-to-commodity mappings
    data = {}
    commodity_data = {}
    
    for category, category_tickers in commodity_dict.items():
        data_category = pd.DataFrame()
        
        for ticker in tqdm(category_tickers, desc = 'Loading Data'):
           
            if ticker in tickers:
                
                # Download historical data for the ticker
                data_ticker = yf.download(ticker, start=start_date, end=end_date)
                
                # Select only the "Adj Close" column and rename it with the commodity name
                commodity_name = ticker_to_commodity.get(ticker, ticker)  # Use the commodity name or ticker if not found
                data_ticker = data_ticker[['Adj Close']].rename(columns={'Adj Close': commodity_name})

                # Concatenate the data to the main DataFrame for the category
                if data_category.empty:
                    data_category = data_ticker
                else:
                    data_category = pd.concat([data_category, data_ticker], axis=1)
                
                # Store ticker-to-commodity mappings in the dictionary
                commodity_data[ticker] = commodity_name
                
                
         # Fill missing values with the median after extracting the data for the current category
        data_category = data_category.apply(lambda col: col.ffill())

        # Store the data for the category in the data dictionary
        data[category] = data_category
    
    return data, commodity_data, commodity_dict


tickers = [
    
    # Hard Commodities - Energy
    
    "CL=F",  # WTI Crude Oil
    "NG=F",  # Natural Gas
    "HO=F",  # Heating Oil
    "RB=F",  # Gasoline
    "BZ=F",  # Brent Crude Oil

    # # Hard Commodities - Precious Metals
    
    "GC=F",  # Gold
    "SI=F",  # Silver
    "PA=F",  # Palladium
    "PL=F",  # Platinum
    
    # Hard Commodities - Base Metals
    
    "HG=F",  # Copper
    "ZN=F",  # Zinc
    
    # Agriculture Commodities
    
    "ZW=F",  # Wheat
    "ZC=F",  # Corn
    "ZS=F",  # Soybeans
    "CC=F",  # Cocoa
    'ZR=F',  # Rice
    'ZO=F',  # Oats
    'GDK=F', # Milk
    'DY=F',  # Dry Whey
    'CB=F',  # Butter
    'GNF=F', # Non Fat Dry Milk
    
    # Livestock Commodities
    
    "HE=F",  # Lean Hogs
    "LE=F",  # Live Cattle
    "GF=F",  # Feeder Cattle

    # Soft Commodities
    
    "SB=F",  # Sugar
    "KC=F",  # Coffee
    "CT=F",  # Cotton
    
    # ETFs
    
    "DBB",  # Invesco DB Base Metals Fund (eposure to aluminum, zinc, and copper)
    "USO",  # United States Oil Fund, LP (exposure oil)
    "DBA",  # Invesco DB Agriculture Fund (Diversified exposure to agricultural commodities)
    'DBC',  # Invesco DB Commodity Index Tracking Fund
    "GSG",   # iShares S&P GSCI Commodity-Indexed Trust: Broad commodities index exposure.
    "RJI",   # Elements Rogers International Commodity Index-Total Return ETN: Broad exposure to commodities.

]


ticker_to_commodity = { 
    
    "CL=F": "WTI Crude Oil", 
    "NG=F": "Natural Gas", 
    "HO=F": "Heating Oil", 
    "RB=F": "Gasoline", 
    "BZ=F": 'Brent Crude Oil',
    
    "GC=F": "Gold", 
    "SI=F": "Silver", 
    "PA=F": "Palladium", 
    "PL=F": "Platinum",
    
    "HG=F": "Copper", 
    'ZN=F': 'Zinc',
    
    "ZW=F": "Wheat", 
    "ZC=F": "Corn", 
    "ZS=F": "Soybeans", 
    "CC=F": "Cocoa",
    'ZC=F': 'Corn',
    'ZR=F': 'Rice',
    'ZO=F': 'Oats',
    'GDK=F': 'Milk',
    'DY=F': 'Dry Whey',
    'CB=F': 'Butter',
    'GNF=F': 'Non Fat Dry Milk',
    
    "HE=F": "Lean Hogs", 
    "LE=F": "Live Cattle", 
    "GF=F": "Feeder Cattle",
    
    "SB=F": "Sugar",
    "KC=F": "Coffee",
    "CT=F": "Cotton",
     
    "DBB":  'Invesco DB Base Metals Fund (exposure to aluminum, zinc, and copper)',
    "USO":  'United States Oil Fund, LP (exposure oil)',
    "DBA":  'Invesco DB Agriculture Fund (Diversified exposure to agricultural commodities)',
    'DBC':  'Invesco DB Commodity Index Tracking Fund',
    "GSG":  'iShares S&P GSCI Commodity-Indexed Trust: Broad commodities index exposure',
    "RJI":  'Elements Rogers International Commodity Index-Total Return ETN: Broad exposure to commodities'
    
}

#%% Calculate Descriptive Statistics and Summary Plots

def compute_returns(data: dict, frequency: str) -> dict:
    returns_data = {}

    for category, category_data in data.items():
        returns_category = pd.DataFrame()
    
        for column_name, column_data in category_data.items():
            
            if frequency == "daily":
                # Compute daily percentage change
                returns = column_data.pct_change()
            
            elif frequency == "monthly":
                # Compute monthly percentage change
                returns = column_data.pct_change().resample("M").ffill()
            
            elif frequency == "yearly":
                # Compute yearly percentage change
                returns = column_data.pct_change().resample("Y").ffill()
            
            else:
                raise ValueError("Invalid frequency. Options: 'daily', 'monthly', 'yearly'")
    
            # Concatenate percentage change to the category DataFrame
            if returns_category.empty:
                returns_category = returns
            else:
                returns_category = pd.concat([returns_category, returns], axis=1)
        
        # Store percentage change in the dictionary
        returns_data[category] = returns_category

    returns_df = pd.concat(returns_data.values(), axis=1)
    
    return returns_data, returns_df



def calculate_covariance_matrix(returns_data: dict) -> pd.DataFrame:

    # Combine log return DataFrames into a single DataFrame
    combined_returns = pd.concat(returns_data.values(), axis=1)

    # Calculate the covariance matrix
    covariance_matrix = combined_returns.cov()

    return covariance_matrix


def calculate_correlation_matrix(returns_data: dict) -> pd.DataFrame:

    # Combine log return DataFrames into a single DataFrame
    combined_returns = pd.concat(returns_data.values(), axis=1)

    # Calculate the correlation matrix
    correlation_matrix = combined_returns.corr()

    return correlation_matrix




# Aggregating metrics
def calculate_mean_std(data):
    
    results = {}

    for category, category_data in data.items():
        
        mean_values = category_data.mean(axis=1)
        std_values = category_data.std(axis=1)
        
        results[category] = {
            'Mean': mean_values,
            'Std': std_values
        }

    return results


# Plotting bell curves
def plot_bell_curves(returns, title='Bell Curves of Returns for Commodity Categories', figsize=(12, 8)):
    
    results = calculate_mean_std(returns)
    plt.figure(figsize=figsize)

    for category, values in results.items():
        
        mean = values['Mean'].mean()
        std = values['Std'].mean()

        x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
        p = np.exp(-(x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

        # Plot the density curve with shading
        plt.plot(x, p, label=f'{category}', linewidth=2)
        plt.fill_between(x, 0, p, alpha=0.2)  # Add shading under the curve

        # Highlight the mean with a vertical line
        plt.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.title(title, fontsize=18)
    plt.xlabel('Returns', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure the bell curve touches the x-axis
    plt.ylim(bottom=0)
    
    plt.show()



#%% Define Portfolio Metrics and additional functions



# Define a function to calculate the maximum drawdown
def calculate_max_drawdown(returns: pd.Series) -> float:
    cum_returns = (1 + returns).cumprod()
    peaks = cum_returns.cummax()
    drawdowns = (cum_returns - peaks) / peaks
    max_drawdown = drawdowns.min()
    return max_drawdown


# Define a function to calculate the sortino ratio
def calculate_sortino_ratio(frequency: str, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sortino ratio for a series of returns.
    
    :param frequency: The frequency of the returns ('daily', 'monthly', 'yearly')
    :param returns: A pandas Series of returns
    :param risk_free_rate: The risk-free rate, default is 0
    :return: The annualized Sortino ratio
    """
    
    # Determine the annualization factor based on the frequency of the returns
    if frequency == 'daily':
        annualization_factor = 252
    elif frequency == 'monthly':
        annualization_factor = 12
    elif frequency == 'yearly':
        annualization_factor = 1
    else:
        raise ValueError("Invalid frequency. Options: 'daily', 'monthly', 'yearly'")
    
    # Adjust risk-free rate for the calculation
    adjusted_risk_free_rate = risk_free_rate / annualization_factor
    
    # Calculate the downside returns
    downside_returns = np.minimum(returns - adjusted_risk_free_rate, 0)
    
    # Calculate the downside volatility (annualized)
    downside_volatility = np.std(downside_returns, ddof=1) * np.sqrt(annualization_factor)
    
    # Calculate and return the annualized Sortino ratio
    annualized_excess_return = np.mean(returns - adjusted_risk_free_rate) * annualization_factor
    sortino_ratio = annualized_excess_return / downside_volatility
    
    return sortino_ratio



# Define a function to calculate the downside deviation
def calculate_downside_deviation(returns: pd.Series, risk_free_rate: float = 0, annualize: bool = False, frequency: str = 'daily') -> float:
    """
    Calculate downside deviation, optionally annualizing the result.
    
    :param returns: Returns as a pandas Series.
    :param risk_free_rate: The minimum acceptable return, often set to the risk-free rate.
    :param annualize: Whether to annualize the downside deviation.
    :param frequency: The frequency of the returns ('daily', 'monthly', 'weekly').
    :return: Downside deviation, annualized if specified.
    """
    # Calculate downside returns
    downside_returns = np.minimum(returns - risk_free_rate, 0)
    
    # Compute downside deviation
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
    
    # Annualize if specified
    if annualize:
        annualization_factors = {'daily': np.sqrt(252), 'weekly': np.sqrt(52), 'monthly': np.sqrt(12)}
        if frequency not in annualization_factors:
            raise ValueError("Invalid frequency. Options: 'daily', 'weekly', 'monthly'")
        downside_deviation *= annualization_factors[frequency]
    
    return downside_deviation



# Define a function to calculate the var
def calculate_var(returns: pd.DataFrame, confidence_level: float = 0.95) -> float:
    
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level should be between 0 and 1")

    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    
    return -var

# Define a function to calculate the cvar
def calculate_cvar(returns: pd.DataFrame, confidence_level: float = 0.95) -> float:
    var = calculate_var(returns, confidence_level)
    sorted_returns = np.sort(returns)
    cvar = sorted_returns[sorted_returns <= var].mean()
    
    return -cvar


# Function for Series Returns
def calculate_portfolio_series_returns(selected_securities: list, weights: pd.Series, prices_data: pd.DataFrame, test_start_index: int) -> pd.Series:
    '''
    Calculate portfolio time series returns based on selected securities, weights, and historical price data.

    Inputs:
    - selected_securities: List of selected securities (column names in prices_data).
    - weights: Pandas Series representing the weights of each security in the portfolio.
    - prices_data: Pandas DataFrame containing historical prices of securities.
    - test_start_index: Index to specify the start of the testing period in prices_data.

    Returns:
    - Pandas Series representing the portfolio's time series returns.

    '''

    # Calculate the corresponding start date based on test_start_index
    start_date = prices_data.index[test_start_index]

    # Filter prices_data for selected securities and start date
    selected_prices = prices_data.loc[start_date:, selected_securities]

    # Calculate portfolio value based on selected securities and weights
    portfolio_value = (selected_prices * weights).sum(axis=1)

    return portfolio_value

# Function for rebalancing
def determine_rebalance_periods(dates, frequency):
    """
    Determines the rebalance periods based on the specified frequency.
    :param dates: Index of the DataFrame, which contains the dates.
    :param frequency: Rebalancing frequency ('monthly', 'yearly', or 'six_months').
    :return: A list of tuples, each tuple represents a period (start_index, end_index).
    """
    periods = []
    trading_days_per_period = {
        'monthly': 21,
        'yearly': 252,
        'six_months': 126
    }

    if frequency not in trading_days_per_period:
        raise ValueError("Frequency must be 'monthly', 'yearly', or 'six_months'.")

    period_length = trading_days_per_period[frequency]
    start_index = 0

    while start_index < len(dates):
        end_index = min(start_index + period_length - 1, len(dates) - 1)
        periods.append((start_index, end_index))
        start_index = end_index + 1

    return periods

#%% Hierarchical Risk Parity Approach

'''
Hierarchical Risk Parity (Prado, 2016) is a portfolio optimization technique that aims to allocate risk in a balanced way across different assets. 
It is a more advanced version of the traditional Risk Parity approach, which equally distributes risk across assets based on their volatility. In HRP, the assets 
are grouped hierarchically based on their correlation structure. 

The algorithm first constructs a dendrogram, which is a tree-like diagram that shows the hierarchical relationships between assets based on their correlation 
matrix. Then, it uses the dendrogram to partition the assets into clusters. The idea behind this approach is that assets within the same cluster are 
expected to have similar risk profiles and therefore should be allocated similar weights. 

At the same time, assets in different clusters are expected to have low correlation with each other and thus, should be allocated different weights. 
HRP is a popular portfolio optimization technique because it can lead to better risk-adjusted returns compared to traditional approaches, 
especially in periods of high market volatility.
 
  The development of this methodology is motivated by some issues regarding widely used strategies, such as:
 
  - Markowitz dependency on quadratic optimization of forecasted returns, frequently providing unstable and highly concentrated solutions.
  - Traditional risk parity ignorance of useful covariance information
This approach is composed of the following steps:
     
    - Tree clustering: group similar investments into clusters based on their correlation matrix. Having a hierarchical structure helps us to improve stability 
issues of quadratic optimizers when inverting the covariance matrix;

    - Quasi-diagonalization: reorganize the covariance matrix so similar investments will be placed together. This matrix diagonalization allow us to distribute
weights optimally following an inverse-variance allocation;

    - Recursive bisection: distribute the allocation through recursive bisection based on cluster covariance.
   
  The goal here is to allocate weights to the assets within each cluster such that the risk contribution of each asset is equal.
   
  Paper available here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678

'''

# Perform Tree Clustering
def tree_clustering(returns_data: pd.DataFrame, method: str, max_clusters: int, window_years: int, frequency: str):
     
    '''
    Parameters:
        
        - returns_data: A DataFrame containing historical returns of securities.
        - selected_securities: A list of selected securities for clustering.
        - method: The linkage method used in hierarchical clustering 
        - max_clusters: The maximum number of clusters to form.
        - window_years: Number of years of historical data to consider.
        - frequency: Frequency of the data
    
    Output:
        
        - It performs hierarchical clustering on the selected securities based on their returns.
        - The resulting clusters are used to create a dendrogram plot (optional).
        - Returns the standardized covariance matrix, cluster labels, and the hierarchical clustering linkage matrix.
    
    '''
    train_data = returns_data.dropna()                           
    
    # Cov Matrix on the training data
    cov_matrix = train_data.cov()
    
    # Convert covariance matrix to distance matrix
    dist_matrix = np.sqrt((1 - cov_matrix.corr(method='pearson')) / 2)
    dist_condensed = pdist(dist_matrix)

    # Perform hierarchical clustering
    Z = linkage(dist_condensed, method=method)

    # Calculate optimal number of clusters using max_clusters parameter
    if max_clusters is None:
        max_clusters = cov_matrix.shape[0]

    # Assign labels to clusters directly without gap threshold
    labels = fcluster(Z, max_clusters, criterion='maxclust')
    
    '''
    ##############################################################################################################################
    #                                                     Dendogram Plot (Opt)                                                   #
    ##############################################################################################################################
    ''' 

    # Plot dendrogram (optional, you can remove this part if you don't need the plot)
    # plt.figure(figsize=(30, 15))
    selected_securities = cov_matrix.columns
    # color_threshold = Z[-max_clusters + 1, 2] if max_clusters > 1 else np.inf
    # dendrogram(Z, orientation='top', labels=selected_securities, above_threshold_color='black',
    #             p=max_clusters, leaf_font_size=20, leaf_rotation=0, color_threshold=color_threshold)
    

    # Change Y ticks
    # y_axis_labels = plt.yticks()[1]
    # for item in y_axis_labels:
    #     item.set_fontsize(20)
    
    # plt.title("Dendrogram", fontsize=15, fontweight='bold')
    # plt.xlabel('Securities', fontsize=15, fontweight='bold')
    # plt.ylabel('Distance', fontsize=15, fontweight='bold')

    # Create dictionary of clusters
    cluster_dict = {}
    for i, cluster_id in enumerate(Z[:, :2]):
        cluster_dict[i] = cluster_id.astype(int)

    cluster_labels = {}
    for i in range(1, max_clusters + 1):
        cluster_labels[i] = list(selected_securities[labels == i])

    return cov_matrix, cluster_labels, dist_matrix

'''
2) Quasi Diagonalization: In the original paper of this algorithm, this step is identified as Quasi-Diagonalisation but it is nothing more than a simple seriation algorithm. 
Matrix seriation is a very old statistical technique which is used to rearrange the data to show the inherent clusters clearly. Using the order of hierarchical 
clusters from the previous step, we rearrange the rows and columns of the covariance matrix of stocks so that similar investments are placed together and dissimilar 
investments are placed far apart. 

This rearranges the original covariance matrix of stocks so that larger covariances are placed along the diagonal and smaller ones around 
this diagonal and since the off-diagonal elements are not completely zero, this is called a quasi-diagonal covariance matrix.

'''

# Perform Quasi Diagonalization
def seriation(Z,N,cur_index):
    '''
    Parameters:
    
    - Z: The linkage matrix resulting from a hierarchical clustering.
    - N: The number of original observations fed into the clustering.
    - cur_index: The current index in the linkage matrix being examined.
    
    Returns:
    - order: A list of indices representing the optimal order.
    
    '''
    if cur_index < N:
        return [cur_index]
    
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
    
    
# Serial Matrix    
def compute_serial_matrix(dist_mat, labels, method:str):
    
    '''
    Parameters:
    
    - dist_mat: numpy.ndarray, the square symmetric distance matrix among elements.
    - labels: list, the names of elements corresponding to the rows/columns of dist_mat.
    - method: str, the linkage method to use. Options include "ward", "single", "average", "complete".
    
    Returns:
    
    - seriated_dist: numpy.ndarray, the input distance matrix, but with re-ordered rows and columns according to the seriation.
    - res_labels: list, the names of elements in the order implied by the hierarchical tree.
    - res_linkage: numpy.ndarray, the hierarchical tree (dendrogram) as a linkage matrix.
    
    '''

    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)

    # Rearrange labels and distance matrix according to the new order
    res_labels = [labels[i] for i in res_order]

    # Initialize a new matrix to hold the seriated distances
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    
    # Apply the new order to rearrange the distances
    seriated_dist[a, b] = dist_mat[np.ix_(res_order, res_order)][a, b]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_labels, res_linkage


'''
3) Recursive Bisection: This step computes HRP portfolio weights given the quasi-diagonal covariance matrix and the order of hierarchical 
clusters. It iteratively allocates weights within clusters based on the variance contribution of each cluster.

    Decision Variables: The weights of assets within each cluster.

    Objective Function: Minimize the difference between the actual risk contributions and the target
    risk contribution (which is the cluster risk contribution divided by the number of assets in the cluster).

    Constraints: Ensure the weights sum up to 1 and are non-negative.
   
'''


def compute_HRP_weights(covariances, res_order, allow_short_selling=False, transaction_cost_rate= None):
    '''
    Parameters:
    - covariances: pandas DataFrame, the covariance matrix of asset returns. The DataFrame should be square, with both rows and columns 
    representing assets.
    - res_order: list or array-like, the order of assets as determined by hierarchical clustering.
    - allow_short_selling: bool, whether to allow short selling in the weight allocation.

    Output:
    - weights: pandas Series, the calculated weights for each asset in the portfolio, indexed by asset identifiers.
    '''
    
    # Initialize weights for each asset to 1
    weights = pd.Series(1, index=res_order)
    # Initialize the list of clusters with the entire set of assets
    clustered_alphas = [res_order]

    # Continue splitting and allocating until all clusters have been processed
    while len(clustered_alphas) > 0:
        # Split each cluster into subclusters, then iterate over them in pairs
        clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                            for start, end in ((0, len(cluster) // 2),
                                               (len(cluster) // 2, len(cluster)))
                            if len(cluster) > 1]
        
        
        for subcluster in range(0, len(clustered_alphas), 2):
            # Extract subclusters
            left_cluster = clustered_alphas[subcluster]
            right_cluster = clustered_alphas[subcluster + 1]

            # Calculate variance for left subcluster
            left_subcovar = covariances.loc[left_cluster, left_cluster]
            inv_diag = 1 / np.diag(left_subcovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            left_cluster_var = np.dot(parity_w, np.dot(left_subcovar.values, parity_w))

            # Calculate variance for right subcluster
            right_subcovar = covariances.loc[right_cluster, right_cluster]
            inv_diag = 1 / np.diag(right_subcovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            right_cluster_var = np.dot(parity_w, np.dot(right_subcovar.values, parity_w))
            
            if allow_short_selling:
                # Allow short selling
                alloc_factor = (right_cluster_var - left_cluster_var) / (left_cluster_var + right_cluster_var)
                weights[left_cluster] *= 0.5 * (1 + alloc_factor)
                weights[right_cluster] *= 0.5 * (1 - alloc_factor)
            else:
                # No short selling
                alloc_factor = left_cluster_var / (left_cluster_var + right_cluster_var)
                weights[left_cluster] *= alloc_factor
                weights[right_cluster] *= 1 - alloc_factor

    return weights



'''
Why Negative Weights Are Not Produced

    Positive Allocations by Design: The Recursive Bisection step divides the portfolio into parts based on variance, where each part is assigned a positive portion of the total allocation. This division is proportional to the inverse of their variance, ensuring that all weights are positive.

    Allocation Factors: In the Recursive Bisection, the allocation between any two clusters (or assets at the final level) is determined by their relative risk. The formula used for dividing the allocation ensures that the weights are between 0 and 1. Specifically, an allocation factor is calculated based on the variances of the left and right clusters, ensuring that each cluster receives a proportion of the portfolio that adds up to 100%. There's no subtraction or mechanism that would allocate a negative weight.

    Cumulative Nature of Allocation: As the allocation process is cumulative (each step divides the remaining portfolio among the next level of clusters or assets), and since the process starts with 100% of the portfolio to be allocated, there's no step at which allocating a negative portion to any asset or cluster would be mathematically justified or necessary.


Utilização de quadractic programming após derivar os weights por HRP?


'''


#%% Backtest Hierarchical Risk Parity

def hrp_portfolios(returns_data: pd.DataFrame, 
                   num_portfolios: int, 
                   num_securities: int, 
                   window_years: int, 
                   frequency: str,
                   method: str,
                   max_clusters: str,
                   allow_short_selling: bool,
                   seed: int,
                   rebalance_frequency: str) -> dict:
    

    np.random.seed(seed)
    
    portfolios_data = {}
    
    # Create a DataFrame to store results for all portfolios
    all_results_dfs = []
    
        
    # Store the updated weights for each rebalance period
    hrp_weights_history = {}

    for i in tqdm(range(num_portfolios), desc='Generating Portfolios (HRP)'):
        
        
        selected_securities = np.random.choice(list(returns_data.columns), size=num_securities, replace=False).tolist()
        
        # Use the frequency parameter to determine the annualization factor
        if frequency == 'daily':
            annualization_factor = 252
        elif frequency == 'monthly':
            annualization_factor = 12
        elif frequency == 'yearly':
            annualization_factor = 1
        else:
            raise ValueError("Invalid frequency. Options: 'daily', 'weekly', 'monthly', 'yearly'")
            
    
        test_start_index = - window_years * annualization_factor
        train_data = returns_data.iloc[:test_start_index][selected_securities]
        test_data = returns_data.iloc[test_start_index:][selected_securities]

        # Portfolio Optimization
        test_cov_matrix = test_data.cov() * annualization_factor
        
        '''
        ##############################################################################################################################
        #                                                     Rebalancing Period                                                     #
        ##############################################################################################################################
        
        1) Initial Train Data Setup:

        Initially, train_data is set to a portion of returns_data that excludes the most recent window_years of data. This initial training dataset is used 
        for the first rebalancing period.

        2) Updating Train Data for Each Rebalancing Period:

        Within the rebalancing loop (for period_start, period_end in rebalance_periods), after the portfolio weights are calculated and applied to the 
        current period's data (period_data), there is an important line at the end of the loop:

            train_data = returns_data.iloc[period_start:test_start_index][selected_securities]

        This line updates the train_data to include data from the start of the current rebalancing period (period_start) up to the beginning of the test set
        (test_start_index). It excludes the data that is being used in the current and future test (or rebalancing) periods.

        3) Implications of Rolling Window Approach:

        The implication of this approach is that each time the portfolio is rebalanced, the training data used for HRP optimization consists of the most 
        recent historical data available up to the start of the current rebalancing period.
    
        '''
        current_portfolio_weights_history = {}
    
        if rebalance_frequency:
            
            rebalance_periods = determine_rebalance_periods(test_data.index, rebalance_frequency)
            portfolio_series = pd.Series(index=test_data.index, dtype=float)
                        
            # Initially, set the end of the previous period to None
            previous_period_end = None
        
            for period_start, period_end in rebalance_periods:
                
                period_data = test_data.iloc[period_start:period_end + 1]
                
                # If there's a previous period, update train_data to include its data, adjusting for window size
                if previous_period_end is not None:
                    
                    # Calculate the number of new rows to add from the period_data
                    new_rows_count = period_end - previous_period_end
                    
                    # Remove the oldest rows to make space for the new period's data
                    train_data = train_data.iloc[new_rows_count:]
                    
                    # Include the new period's data into train_data
                    train_data = pd.concat([train_data, period_data])
                    
                
                # Call HRP Functions
                cov_matrix, cluster_labels, dist_matrix = tree_clustering(train_data, method=method, max_clusters = max_clusters, window_years=window_years, frequency=frequency)
                ordered_dist_mat, res_labels, res_linkage = compute_serial_matrix(dist_matrix.values, selected_securities, method=method)
                weights = compute_HRP_weights(cov_matrix, res_labels, allow_short_selling)
    
                # Calculate portfolio returns for the period
                period_returns = period_data.dot(weights)
                portfolio_series.iloc[period_start:period_end+1] = period_returns
                
                # Update train_data for the next period by including current period_data
                previous_period_end = period_end
                
                # Here, we capture the weights after each rebalancing
                rebalance_date = test_data.index[period_start]
                
                current_portfolio_weights_history[rebalance_date] = weights

        else:
            # Call HRP Functions
            cov_matrix, cluster_labels, dist_matrix = tree_clustering(train_data, method=method, max_clusters = max_clusters, window_years=window_years, frequency=frequency)
            ordered_dist_mat, res_labels, res_linkage = compute_serial_matrix(dist_matrix.values, selected_securities, method=method)
            weights = compute_HRP_weights(cov_matrix, res_labels, allow_short_selling)
            portfolio_series = test_data.dot(weights)
            
        # Store the current portfolio's weights history in the main dictionary
        portfolio_name = f'Portfolio_{i + 1}'
        hrp_weights_history[portfolio_name] = current_portfolio_weights_history
        
        '''
        ##############################################################################################################################
        #                                                     Performance Metrics                                                    #
        ##############################################################################################################################
        ''' 
        
        
        # Call calculate_portfolio_series_returns to get portfolio series returns
        portfolio_series_returns = calculate_portfolio_series_returns(selected_securities=selected_securities,
                                                                      weights=weights, prices_data=test_data,
                                                                      test_start_index=test_start_index).rename('Returns')
        
        cumulative_returns = (1 + portfolio_series_returns).cumprod().rename('Cumulative Returns')
        
        
        
        # Calculate portfolio returns for the test set
        
        total_periodic_returns_sum = np.sum(test_data.dot(weights))
        num_periods = len(test_data)
        
        if frequency == 'daily':
            periods_per_year = 252
        elif frequency == 'monthly':
            periods_per_year = 12
        elif frequency == 'yearly':
            periods_per_year = 1
        else:
            raise ValueError("Invalid frequency. Options: 'daily', 'monthly', 'yearly'")
        
        annualized_return = (total_periodic_returns_sum / num_periods) * periods_per_year

        # Calculate portfolio volatility for the test set (standard deviation)
        portfolio_volatility_test = np.sqrt(np.dot(weights.T, np.dot(test_cov_matrix, weights))).round(5)

        # Calculate portfolio Sharpe ratio for the test set
        sharpe_ratio_test = (annualized_return / portfolio_volatility_test).round(5)
        
        # Calculate portfolio Sortino ratio for the test set
        sortino_ratio_test = round(calculate_sortino_ratio('daily',test_data.dot(weights)), 5)

        # Calculate maximum drawdown for the test set
        max_drawdown_test = round(calculate_max_drawdown(test_data.dot(weights)), 5)
        
        # Calculate downside deviation for the test set
        downside_deviation_test = round(calculate_downside_deviation(test_data.dot(weights)), 5)
        
        # Calculate var for the test set
        var_test = round(calculate_var(test_data.dot(weights)), 5)
        
        # Calculate cvar for the test set
        cvar_test = round(calculate_cvar(test_data.dot(weights)), 5)
        

        '''
        ##############################################################################################################################
        #                                                     Storing Results                                                        #
        ##############################################################################################################################
        ''' 
        
        # Create a new DataFrame for each portfolio
        results_df = pd.DataFrame(columns=['Portfolio', 'Returns', 'Volatility', 'Sharpe Ratio', 
                                           'Max Drawdown', 'Sortino Ratio', 'Downside Deviation', 'VAR', 'CVAR'])

        
        # Append results to the DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({
            
            'Portfolio': [f'Portfolio_{i + 1}'],
            'Returns': [annualized_return.round(5)],
            'Volatility': [portfolio_volatility_test],
            'Sharpe Ratio': [sharpe_ratio_test],
            'Sortino Ratio': [sortino_ratio_test],
            'Max Drawdown': [max_drawdown_test],
            'Downside Deviation': [downside_deviation_test],
            'VAR': [var_test],
            'CVAR': [cvar_test]
            
        })], ignore_index=True)
        
        
        # Append the results_df to the list
        all_results_dfs.append(results_df)


        # Store portfolio data in the dictionary
        portfolio_name = f'Portfolio_{i + 1}'
        
        portfolios_data[portfolio_name] = {
            
            'Weights': weights,
            'Portfolio Series': cumulative_returns,
            'Portfolio Stats': results_df,
        }
        
         
    '''
    ##############################################################################################################################
    #                                                     Performing Final Adjustments                                           #
    ##############################################################################################################################
    '''

    # Combine all results DataFrames into a single DataFrame
    final_results_df = pd.concat(all_results_dfs, ignore_index=True)

    # Calculate the median of the final DataFrame
    numeric_columns = final_results_df.select_dtypes(include=np.number).columns
    median_results = final_results_df[numeric_columns].median().to_frame().T
    median_results['Strategy'] = 'Hierarchical Risk Parity'
    median_results.set_index('Strategy', inplace=True)
    
    return final_results_df, median_results, portfolios_data, hrp_weights_history



#%%

## Importing Data
start_date, end_date = get_dates()
data, commodity_data, commodity_dict = import_commodity_data(tickers, start_date, end_date)


## Storing data into dataframe
all_data_df = pd.DataFrame()
 
for category, df in data.items():
    if all_data_df.empty:
        all_data_df = df
    else:
        all_data_df = pd.concat([all_data_df, df], axis=1)


# Performance Metrics
returns, returns_df_daily = compute_returns(data = data, frequency = 'daily')
returns_monthly, returns_df_monthly = compute_returns(data = data, frequency = 'monthly')
correlation_matrix = calculate_correlation_matrix(returns_data = returns)
plot_bell_curves(returns_monthly, title='Empirical Distributions of Monthly Commodity Futures', figsize=(14, 10))



# HRP Optimization
hrp_portfolios_results, hrp_strategies_final_results, hrp_portfolio_dict, hrp_weights_history = hrp_portfolios(
    
                                                                                        returns_data = returns_df_daily, 
                                                                                        num_portfolios = 100, 
                                                                                        num_securities = 10, 
                                                                                        window_years = 3,
                                                                                        frequency = 'daily',
                                                                                        method = 'ward',
                                                                                        max_clusters=5,
                                                                                        allow_short_selling = False,
                                                                                        seed = 123,
                                                                                        rebalance_frequency='six_months'
                                                                                        
                                                                                        )

hrp_portfolios_results1, hrp_strategies_final_results1, hrp_portfolio_dict1, hrp_weights_history1 = hrp_portfolios(
    
                                                                                        returns_data = returns_df_daily, 
                                                                                        num_portfolios = 100, 
                                                                                        num_securities = 10, 
                                                                                        window_years = 3,
                                                                                        frequency = 'daily',
                                                                                        method = 'ward',
                                                                                        max_clusters=5,
                                                                                        allow_short_selling = True,
                                                                                        seed = 123,
                                                                                        rebalance_frequency='six_months'
                                                                                        
                                                                                        )


hrp_portfolios_results2, hrp_strategies_final_results2, hrp_portfolio_dict2, hrp_weights_history2 = hrp_portfolios(
    
                                                                                        returns_data = returns_df_daily, 
                                                                                        num_portfolios = 100, 
                                                                                        num_securities = 10, 
                                                                                        window_years = 5,
                                                                                        frequency = 'daily',
                                                                                        method = 'ward',
                                                                                        max_clusters=5,
                                                                                        allow_short_selling = False,
                                                                                        seed = 123,
                                                                                        rebalance_frequency='six_months'
                                                                                        
                                                                                        )


hrp_portfolios_results3, hrp_strategies_final_results3, hrp_portfolio_dict3, hrp_weights_history3 = hrp_portfolios(
    
                                                                                        returns_data = returns_df_daily, 
                                                                                        num_portfolios = 100, 
                                                                                        num_securities = 10, 
                                                                                        window_years = 5,
                                                                                        frequency = 'daily',
                                                                                        method = 'ward',
                                                                                        max_clusters=5,
                                                                                        allow_short_selling = True,
                                                                                        seed = 123,
                                                                                        rebalance_frequency='six_months'
                                                                                        
                                                                                        )


hrp_portfolios_results4, hrp_strategies_final_results4, hrp_portfolio_dict4, hrp_weights_history4 = hrp_portfolios(
    
                                                                                        returns_data = returns_df_daily, 
                                                                                        num_portfolios = 100, 
                                                                                        num_securities = 10, 
                                                                                        window_years = 7,
                                                                                        frequency = 'daily',
                                                                                        method = 'ward',
                                                                                        max_clusters=5,
                                                                                        allow_short_selling = False, 
                                                                                        seed = 123,
                                                                                        rebalance_frequency='six_months'
                                                                                        
                                                                                        )


hrp_portfolios_results5, hrp_strategies_final_results5, hrp_portfolio_dict5, hrp_weights_history5 = hrp_portfolios(
    
                                                                                        returns_data = returns_df_daily, 
                                                                                        num_portfolios = 100, 
                                                                                        num_securities = 10, 
                                                                                        window_years = 7,
                                                                                        frequency = 'daily',
                                                                                        method = 'ward',
                                                                                        max_clusters=5,
                                                                                        allow_short_selling = True,
                                                                                        seed = 123,
                                                                                        rebalance_frequency='six_months'
                                                                                        
                                                                                        )
