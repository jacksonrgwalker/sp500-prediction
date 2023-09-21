import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.tools import add_constant
from tqdm import tqdm

# Load in the dataframe with OHLC data as well as Fama-French 5-Factor data
ff_file_path = "data/F-F_Research_Data_5_Factors_2x3_daily.CSV"
ohlc_file_path = "data/ohlc.parquet"

ff_dtypes = {
    "Mkt-RF": pd.Float64Dtype(),
    "SMB": pd.Float64Dtype(),
    "HML": pd.Float64Dtype(),
    "RMW": pd.Float64Dtype(),
    "CMA": pd.Float64Dtype(),
    "RF": pd.Float64Dtype(),
}
ff = pd.read_csv(ff_file_path, index_col=0, dtype=ff_dtypes)
ohlc = pd.read_parquet(ohlc_file_path)

# Format Fama-French DataFrame
ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
ff.index.name = "date"
# Convert to decimal format
factor_cols = ff.columns.drop("RF")
ff[factor_cols] = ff[factor_cols] / 100

# Join the FF data and OHLC data
ohlc = ohlc.join(ff, on="date", how="inner")

# Perform rolling OLS with the FF 5-Factor Model data
symbol = ohlc.index.get_level_values("symbol").unique()
rolling_window_length = 50

# sort by date so rolling windows
# and return calcs work properly
ohlc.sort_index(inplace=True)
stocks = ohlc.groupby("symbol")
ohlc["returns"] = stocks["adjusted_close"].pct_change()
ohlc["log_returns"] = np.log(ohlc["adjusted_close"] / stocks["adjusted_close"].shift(1))
ohlc["excess_returns"] = ohlc["returns"] - ohlc["RF"]
ohlc["excess_log_returns"] = ohlc["log_returns"] - np.log(ohlc["RF"])
# remove rows with missing values from returns calculations
ohlc.dropna(inplace=True)

column_names = ["beta_" + factor for factor in factor_cols]
column_names = ["beta_idosyncratic"] + column_names
beta_dfs = {}
progress_bar = tqdm(symbol, desc="Calculating coeffs for stocks")

for symbol in progress_bar:
    # Show current symbol in progress bar
    progress_bar.set_postfix_str(symbol.ljust(5, " "))

    # Get the data for the current stock
    this_stock = ohlc.loc[symbol].copy()

    # Skip if there is not enough data
    if len(this_stock) < rolling_window_length:
        continue

    # Form independent and dependent variable matrices
    X = this_stock[factor_cols].astype(float)
    X = add_constant(X)

    y = this_stock["excess_returns"].astype(float)

    rolling_ols = RollingOLS(
        endog=y, exog=X, window=rolling_window_length, min_nobs=10, expanding=True
    ).fit()

    coefficients = rolling_ols.params
    coefficients.columns = column_names

    # Drop the first rolling_window_length rows since they are all NaNs
    coefficients.dropna(inplace=True)

    coefficients["idosyncratic_change"] = coefficients[
        "beta_idosyncratic"
    ].pct_change(-1)
    coefficients.dropna(inplace=True)

    # Save the coefficients for this stock
    beta_dfs[symbol] = coefficients

beta_df = pd.concat(beta_dfs, axis=0)

# Save the table
beta_save_path = Path("data/beta.parquet")
beta_df.to_parquet(beta_save_path)
