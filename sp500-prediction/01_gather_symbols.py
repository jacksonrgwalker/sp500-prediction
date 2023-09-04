import pandas as pd
from pathlib import Path

# Grab table from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
symbols = pd.read_html(url)[0]

# Clean up the table
symbols.columns = symbols.columns.str.lower().str.replace(" ", "_", regex=False)
symbols["symbol"] = symbols["symbol"].str.strip().str.replace(".", "-", regex=False)
symbols['date_added'] = pd.to_datetime(symbols['date_added'], format='%Y-%m-%d', errors='coerce')

dtypes = {
    "symbol": pd.StringDtype(),
    "security": pd.StringDtype(),
    "gics_sector": pd.CategoricalDtype(),
    "gics_sub-industry": pd.CategoricalDtype(),
    "headquarters_location": pd.CategoricalDtype(),
    "cik": pd.Int64Dtype(),
    "founded": pd.StringDtype()
}

symbols = symbols.astype(dtypes)

# Save the table
symbols_save_path = Path("data/symbols.parquet")
symbols.to_parquet(symbols_save_path)
