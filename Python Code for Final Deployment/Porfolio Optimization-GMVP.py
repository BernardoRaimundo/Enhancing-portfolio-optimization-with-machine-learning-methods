#%% Importing Libraries and initial setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as optimize
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

#%% Global Minimum Variance Portfolio

def optimize_portfolio_for_gmvp(train_data, selected_securities, allow_short_selling, frequency):
    # Define the constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)  # Sum of weights must be 1 (fully invested)

    # Define the bounds based on short selling allowance
    if allow_short_selling:
        bounds = [(-1, 1) for _ in selected_securities]  # Allowing short selling
    else:
        bounds = [(0, 1) for _ in selected_securities]  # No short selling
        
        
    # Initial guess for weights (equal weights for all assets)
    initial_weights = [1. / len(selected_securities)] * len(selected_securities)

    # Calculate annualization factor based on frequency
    annualization_factor = {'daily': 252, 'weekly': 52, 'monthly': 12, 'yearly': 1}.get(frequency)
    if annualization_factor is None:
        raise ValueError("Invalid frequency. Options: 'daily', 'weekly', 'monthly', 'yearly'")
    
    # Covariance matrix of returns, annualized
    cov_matrix = train_data.cov() * annualization_factor

    # Transaction costs model: Linear cost assumption
    def transaction_costs(weights, initial_weights, transaction_cost_rate):
        return np.sum(np.abs(weights - initial_weights)) * transaction_cost_rate

    # Objective function to minimize: Portfolio Variance +/- Transaction Costs
    def objective_function(weights):
        variance = weights.T @ cov_matrix @ weights
        return variance

    # Perform the optimization with constraints
    result = optimize.minimize(objective_function,
                               initial_weights,
                               method='SLSQP',
                               bounds=bounds,
                               constraints=constraints)

    # Extract optimized weights from the result
    optimized_weights = pd.Series(data=result.x, index=selected_securities)
    
    return optimized_weights

#%% Global Minimum Variance Portfolio

# Define GMVP approach
def gmvp(returns_data: pd.DataFrame, 
                             num_portfolios: int, 
                             num_securities: int, 
                             window_years: int, 
                             frequency: str,  
                             allow_short_selling: bool,
                             seed: int,
                             rebalance_frequency: str = None) -> dict:
    
    
    np.random.seed(seed)

    # The function initializes an empty dictionary to store various performance metrics for each generated portfolio.
    portfolios_data = {}
    
    # Create a DataFrame to store results for all portfolios
    all_results_dfs = []
    
    # Store the updated weights for each rebalance period
    mvo_weights_history = {}
    
    
    '''
    ##############################################################################################################################
    #                                                     Splitting the Data                                                     #
    ##############################################################################################################################
    ''' 
    
    for i in tqdm(range(num_portfolios), desc='Generating Portfolios (GMVP)'):
        

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
        current period's data (period_data), the train data is updated to include the most recent information at that time while discarding older infromation.
        
        This is done for each rebalancing window.


        3) Implications of Rolling Window Approach:

        The implication of this approach is that each time the portfolio is rebalanced, the training data used for MVO optimization consists of the most 
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
                    
                
                weights = optimize_portfolio_for_gmvp(train_data, selected_securities, allow_short_selling, frequency)
    
                # Calculate portfolio returns for the period

                period_returns = period_data.dot(weights)
                portfolio_series.iloc[period_start:period_end+1] = period_returns
                
                # Update train_data for the next period by including current period_data
                previous_period_end = period_end
                
                # Here, we capture the weights after each rebalancing
                rebalance_date = test_data.index[period_start]
                
                current_portfolio_weights_history[rebalance_date] = weights
                
        else:
            weights = optimize_portfolio_for_gmvp(train_data, selected_securities, allow_short_selling, frequency)
            portfolio_series = test_data.dot(weights)
            
        # Store the current portfolio's weights history in the main dictionary
        portfolio_name = f'Portfolio_{i + 1}'
        mvo_weights_history[portfolio_name] = current_portfolio_weights_history
            
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
        sortino_ratio_test = round(calculate_sortino_ratio('daily', test_data.dot(weights)), 5)

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
    median_results['Strategy'] = 'Global Minimum Variance Portfolio'
    median_results.set_index('Strategy', inplace=True)
    
    return final_results_df, median_results, portfolios_data, mvo_weights_history


