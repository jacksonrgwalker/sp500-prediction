from pathlib import Path

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm

from dotenv import load_dotenv

assert load_dotenv(), "Failed to load .env file"

# Load symbols table
symbols_save_path = Path("data/symbols.parquet")
symbols = pd.read_parquet(symbols_save_path)
symbols_to_gather = symbols["symbol"].tolist()

# initialize Alpha Vantage API
ts = TimeSeries()

problem_symbols = {
    "FBHS",
    "BMC",
    "TLAB",
    "CBE",
    "DF",
    "EK",
    "LEH",
    "DJ",
    "SLR",
    "JOYG",
    "FRE",
    "ACE",
    "FNM",
    "ABK",
    "SHLD",
    "AV",
    "TRB",
    "MEE",
    "CFC",
    "CCE",
    "ESV",
    "NOVL",
    "WIN",
    "TE",
    "COG",
    "SMS",
    "GR",
    "BS",
    "FB",
    "KFT",
    "FRC",
    "RX",
    "HFC",
    "QTRN",
    "ADS",
    "RE",
    "PTV",
    "MOLX",
    "LDW",
    "NYX",
    "TYC",
    "SBL",
    "TSO",
    "GLK",
    "MI",
    "GENZ",
    "SGP",
    "WLTW",
    "JCP",
    "KSE",
    "ABS",
    "WFR",
    "CEPH",
    "HPH",
}

# Gather OHLC data for each symbol
ohlc_tables = {}
# tqdm for fancy progress bar
progress_bar = tqdm(symbols_to_gather, desc="Gathering stock data")
for symbol in progress_bar:
    # Show current symbol in progress bar
    progress_bar.set_postfix_str(symbol.ljust(5, " "))

    if symbol in problem_symbols:
        continue

    # API call to Alpha Vantage to get daily adjusted OHLC data
    ohlc, _meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize="full")

    # Convert to Pandas DataFrame and clean up column names
    ohlc = pd.DataFrame.from_dict(ohlc, orient="index")
    ohlc.columns = ohlc.columns.str[3:].str.replace(" ", "_")

    ohlc_tables[symbol] = ohlc

# Combine all tables into one
ohlc_all = pd.concat(ohlc_tables)
ohlc_all.index.set_names(["symbol", "date"], inplace=True)
ohlc_all = ohlc_all.reset_index()
ohlc_all["date"] = pd.to_datetime(ohlc_all["date"])
ohlc_all["symbol"] = ohlc_all["symbol"].astype("category")
ohlc_all = ohlc_all.set_index(["symbol", "date"])

# Convert to correct data types
ohlc_dtypes = {
    "open": pd.Float64Dtype(),
    "high": pd.Float64Dtype(),
    "low": pd.Float64Dtype(),
    "close": pd.Float64Dtype(),
    "adjusted_close": pd.Float64Dtype(),
    "volume": pd.Int64Dtype(),
    "dividend_amount": pd.Float64Dtype(),
    "split_coefficient": pd.Float64Dtype(),
}
ohlc_all = ohlc_all.astype(ohlc_dtypes)

# Sort by symbol and date
ohlc_all.sort_index(inplace=True)

# Save the table
ohlc_save_path = Path("data/ohlc.parquet")
ohlc_all.to_parquet(ohlc_save_path)
