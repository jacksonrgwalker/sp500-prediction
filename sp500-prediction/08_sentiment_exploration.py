from glob import glob
import pandas as pd
from pathlib import Path
import numpy as np

# THIS IS NOT AN ACTUAL RUN SCRIPT,
# JUST A REFERENCE FOR THE STEPS TAKEN TO GET THE sentiment_exploration
# DATAFRAME. THIS WILL BE REMOVED LATER

# read in all the parquet files
nyt_files = glob("data/NYT/*/*.parquet")
articles = pd.concat([pd.read_parquet(f) for f in nyt_files])
articles.reset_index(drop=True, inplace=True)

symbols_path = Path("data/symbols.parquet")
symbols = pd.read_parquet(symbols_path)

mentions_filepath = Path("data/mentions.parquet")
mentions = pd.read_parquet(mentions_filepath)

ohlc_path = Path("data/ohlc.parquet")
ohlc = pd.read_parquet(ohlc_path)

embeddings_path = Path("data/nyt_embeddings.parquet")
embeddings = pd.read_parquet(embeddings_path)


mentions = mentions.merge(articles[["pub_date", "_id"]], on="_id", how="left")
mentions["pub_date"] = mentions["pub_date"].dt.date
mentions["pub_date"] = pd.to_datetime(mentions["pub_date"])

# get returns
ohlc["return_today"] = (
    ohlc["adjusted_close"].apply(np.log).groupby("symbol", observed=True).diff()
)
for i in range(1, 8):
    ohlc[f"return_{i:02}_day_prev"] = (
        ohlc["return_today"].groupby("symbol", observed=True).shift(i)
    )
    ohlc[f"return_{i:02}_day_post"] = (
        ohlc["return_today"].groupby("symbol", observed=True).shift(-i)
    )

return_cols = [c for c in ohlc.columns if "return" in c]

mentions = mentions.merge(
    ohlc[return_cols], left_on=["symbol", "pub_date"], right_index=True
)

embeddings = embeddings.rename(columns=lambda x: f"x_{x:04}")
mentions = mentions.merge(embeddings, on="_id", how="left")

mentions.rename(
    columns={"_id": "article_id", "pub_date": "article_published_date"}, inplace=True
)

sentiment_exploration_path = Path("data/sentiment_exploration.parquet")
mentions.to_parquet(sentiment_exploration_path)