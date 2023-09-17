import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.tools import add_constant
from tqdm import tqdm
from copy import deepcopy

# Load in the dataframe with OHLC data as well as Fama-French 5-Factor data
operating_directory = Path.cwd()
ff_file_extension = "data/F-F_Research_Data_5_Factors_2x3_daily.CSV"
ohlc_file_extension = "data/ohlc.parquet"

ff_df = pd.read_csv(operating_directory / ff_file_extension)

ohlc_df = pd.read_parquet(operating_directory / ohlc_file_extension)

# Format Fama-French DataFrame
ff_df[ff_df.columns[0]] = pd.to_datetime(ff_df[ff_df.columns[0]], format='%Y%m%d')
ff_df.set_index(ff_df.columns[0], inplace=True, drop=True)

# Join the FF data and OHLC data
multi_index_names = ohlc_df.index.names
combined_df = ohlc_df.join(ff_df, on=[multi_index_names[1]])

# Perform rolling OLS with the FF 5-Factor Model data
ticker_list = ohlc_df.index.get_level_values(multi_index_names[0]).unique()
rolling_window_length = 50

# Create DataFrame to store 5-Factor coefficients
factors = ['idosynchratic']
factors.extend(ff_df.columns[:-1])
column_names = ['beta_' + factors[i] for i in range(len(factors))]
counter = 1

progress_bar = tqdm(ticker_list, desc="Calculating coeffs for stocks")
for ticker in progress_bar:

    # Show current symbol in progress bar
    progress_bar.set_postfix_str(ticker.ljust(5, " "))

    stock_specific_df = deepcopy(combined_df[combined_df.index.get_level_values(multi_index_names[0]) == ticker])

    stock_specific_df['excess_simple_returns'] = (stock_specific_df['adjusted_close'] \
                                          / stock_specific_df['adjusted_close'].shift(1) - 1.0) \
                                          - stock_specific_df['RF']

    stock_specific_df.dropna(inplace=True)

    # Form independent and dependent variable matrices
    X = add_constant(stock_specific_df[factors[1:]]/100).to_numpy()
    y = np.expand_dims(stock_specific_df['excess_simple_returns'].to_numpy(), axis=1)

    # Fit the linear model and save the coefficients if the stock has enough data
    if (X.shape[0] > rolling_window_length) and (y.shape[0] > rolling_window_length):

        results = RollingOLS(endog=y, exog=X, window=rolling_window_length, min_nobs=10, expanding=True).fit()

        coefficients_df = pd.DataFrame(results.params, index=stock_specific_df.index,
                                       columns=column_names).dropna()

        coefficients_df['idosynchratic_change'] = coefficients_df['beta_idosynchratic'].pct_change(-1)
        coefficients_df.dropna(inplace=True)

        if counter == 1:
            beta_df = deepcopy(coefficients_df)
            counter += 1
        else:
            beta_df = pd.concat([beta_df, coefficients_df])

# Save the table
beta_save_path = Path("data/beta.parquet")
beta_df.to_parquet(beta_save_path)