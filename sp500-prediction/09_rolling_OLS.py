import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.tools import add_constant
from tqdm import tqdm
from copy import deepcopy

# Load in the dataframe with OHLC data as well as Fama-French 5-Factor data
operating_directory = Path.cwd()
FF_file_extension = "data/F-F_Research_Data_5_Factors_2x3_daily.CSV"
OHLC_file_extension = "data/ohlc.parquet"

ff_df = pd.read_csv(operating_directory / FF_file_extension)

ohlc_df = pd.read_parquet(operating_directory / OHLC_file_extension)

# Format Fama-French DataFrame
ff_df[ff_df.columns[0]] = pd.to_datetime(ff_df[ff_df.columns[0]], format='%Y%m%d')
ff_df.set_index(ff_df.columns[0], inplace=True, drop=True)

# Perform rolling OLS with the FF 5-Factor Model data
multi_index_names = ohlc_df.index.names
ticker_list = ohlc_df.index.get_level_values(multi_index_names[0]).unique()
rolling_window_length = 50

# Create DataFrame to store 5-Factor coefficients
column_names = ['beta_' + str(i) for i in range(6)]
counter = 1

# The last date in the FF data file
max_ff_date = np.max(ff_df.index)

progress_bar = tqdm(ticker_list, desc="Calculating coeffs for stocks")
for ticker in progress_bar:

    # Show current symbol in progress bar
    progress_bar.set_postfix_str(ticker.ljust(5, " "))

    # Build a DataFrame for simple returns for the current stock and for the correct dates
    returns_df = ohlc_df[(ohlc_df.index.get_level_values(multi_index_names[0])
                                    == ticker) & (ohlc_df.index.get_level_values(multi_index_names[1])
                                    <= max_ff_date)]['adjusted_close'].to_frame()

    returns_df['simple_return'] = (returns_df['adjusted_close']
                                         / returns_df['adjusted_close'].shift(1) - 1)

    # Reduce the FF data to only the dates contained within the current stock
    ticker_min_date = np.min(returns_df.index.get_level_values(multi_index_names[1]))
    ticker_max_date = np.max(returns_df.index.get_level_values(multi_index_names[1]))
    timed_FF_df = ff_df[(ff_df.index >= ticker_min_date) & (ff_df.index <= ticker_max_date)]

    # Join the FF data and returns DataFrame to efficiently remove NA from both datasets before
    # OLS fitting
    fit_df = returns_df.join(timed_FF_df, on=[multi_index_names[1]]).iloc[:,1:].dropna()
    X = add_constant(fit_df.iloc[:,1:-1]).to_numpy()
    y = fit_df.iloc[:,0].to_numpy()

    # Fit the linear model and save the coefficients
    results = RollingOLS(endog=y, exog=X, window=rolling_window_length).fit()

    coefficients_df = pd.DataFrame(results.params, index=fit_df.index,
                                   columns=column_names).dropna()

    if counter == 1:
        beta_df = deepcopy(coefficients_df)
        counter += 1
    else:
        beta_df = pd.concat([beta_df, coefficients_df])

# Save the table
beta_save_path = Path("data/beta.parquet")
beta_df.to_parquet(beta_save_path)