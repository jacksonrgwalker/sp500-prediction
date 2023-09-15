import pandas as pd
from pathlib import Path

# Grab table from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
wiki_tables = pd.read_html(url)
symbols = wiki_tables[0].copy()

# Clean up the table
symbols.columns = (
    symbols.columns.str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)
symbols["symbol"] = symbols["symbol"].str.strip().str.replace(".", "-", regex=False)
symbols["date_added"] = pd.to_datetime(
    symbols["date_added"], format="%Y-%m-%d", errors="coerce"
)

old_symbols = wiki_tables[1].copy()
old_symbols.columns = [
    "date",
    "added_ticker",
    "added_security",
    "removed_ticker",
    "removed_security",
    "reason",
]
old_symbols.columns = old_symbols.columns.str.lower().str.replace(" ", "_", regex=False)
old_symbols["date"] = pd.to_datetime(old_symbols["date"])

removed = old_symbols[["removed_ticker", "removed_security", "date"]].copy().dropna()
added = old_symbols[["added_ticker", "added_security", "date"]].copy().dropna()
removed.columns = ["symbol", "security", "date_removed"]
added.columns = ["symbol", "security", "date_added"]


# if symbol has > 1 date_removed, keep the latest date removed
removed = removed.sort_values(by=["date_removed"]).drop_duplicates(
    subset=["symbol"], keep="last"
)

# if symbol has > 1 date_added, keep the earliest date added
added = added.sort_values(by=["date_added"]).drop_duplicates(
    subset=["symbol"], keep="first"
)

old_symbols = added.merge(
    removed, on=["symbol"], how="outer", suffixes=("", "_removed")
)
old_symbols["security"].fillna(old_symbols["security_removed"], inplace=True)
old_symbols.drop(columns=["security_removed"], inplace=True)

all_symbols = symbols.merge(
    old_symbols, on=["symbol"], how="outer", suffixes=("", "_old")
)
all_symbols["security"].fillna(all_symbols["security_old"], inplace=True)
all_symbols.drop(columns=["security_old"], inplace=True)
all_symbols["date_added"].fillna(all_symbols["date_added_old"], inplace=True)
all_symbols.drop(columns=["date_added_old"], inplace=True)

removed_before_2000 = all_symbols["date_removed"] < "2000-01-01"
unknown_date_removed = all_symbols["date_removed"].isna()
all_symbols = all_symbols[(~removed_before_2000) | unknown_date_removed]

dtypes = {
    "symbol": pd.StringDtype(),
    "security": pd.StringDtype(),
    "gics_sector": pd.CategoricalDtype(),
    "gics_sub_industry": pd.CategoricalDtype(),
    "headquarters_location": pd.CategoricalDtype(),
    "cik": pd.Int64Dtype(),
    "founded": pd.StringDtype(),
}

all_symbols = all_symbols.astype(dtypes)

assert 'MMM' in all_symbols['symbol'].values, "MMM not in all_symbols"
assert len(all_symbols) > 500, "all_symbols has less than 500 rows"

# Save the table
symbols_save_path = Path("data/symbols.parquet")
all_symbols.to_parquet(symbols_save_path)
