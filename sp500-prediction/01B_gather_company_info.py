from pathlib import Path

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from requests.exceptions import HTTPError
from tqdm import tqdm

# Load symbols table
symbols_save_path = Path("data/symbols.parquet")
symbols = pd.read_parquet(symbols_save_path)
symbols_to_gather = symbols["symbol"].tolist()


company_info_tables = {}
# tqdm for fancy progress bar
progress_bar = tqdm(symbols_to_gather, desc="Gathering company info")
for symbol in progress_bar:
    # Show current symbol in progress bar
    progress_bar.set_postfix_str(symbol.ljust(5, " "))

    yf_ticker = yf.Ticker(symbol)

    try:
        this_company_info = pd.json_normalize(yf_ticker.info)
    except HTTPError as e:
        if e.response.status_code == 404:
            continue
        else:
            raise e

    company_info_tables[symbol] = this_company_info

# Combine all tables into one
company_info_all = pd.concat(company_info_tables)
company_info_all = company_info_all.convert_dtypes(dtype_backend='pyarrow')
company_info_all.reset_index(level=1, drop=True, inplace=True)
company_info_all.index.name = 'symbol'
assert (company_info_all.index == company_info_all['symbol']).all(), "Searched symbol and returned symbol do not match"
company_info_all.drop(columns=['symbol'], inplace=True)

# fix stubborn data types
company_info_all = company_info_all.astype({
    'trailingPE' : pd.Float64Dtype(),
    'forwardPE' : pd.Float64Dtype(),
})

# Save the table
company_info_save_path = Path("data/company_info.parquet")
company_info_all.to_parquet(company_info_save_path)
