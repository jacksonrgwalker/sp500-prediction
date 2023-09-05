import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from requests.adapters import Retry
from tqdm import tqdm

assert load_dotenv(), "Failed to load .env file"

# Params ################
MAX_API_CALLS_PER_MINUTE = 190 
NO_RESULT_LIMIT = 3 # number of months with no results before skipping symbol
#########################

# calculate the min delay between api calls 
# so we don't exceed the rate limit
min_delay_seconds = 60/MAX_API_CALLS_PER_MINUTE

# Get the Alpha Vantage API key from the environment variable
api_key = os.environ["ALPHAVANTAGE_API_KEY"]

# Set the API endpoint URL with the apikey parameter
url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&time_from={time_from}&time_to={time_to}&limit=1000&apikey={api_key}"

# Create a list of symbols and years
symbols = pd.read_parquet("data/symbols.parquet")["symbol"].tolist()

# Create a list of years and months
years = list(range(2000, 2024))
months = list(range(1, 13))
year_months = [(year, month) for year in years for month in months]
year_months = year_months[::-1]

# Define the retry strategy
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)

# Create a session with the retry adapter
session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry_strategy))

dtypes = {
    "title": pd.StringDtype(),
    "url": pd.StringDtype(),
    "summary": pd.StringDtype(),
    "banner_image": pd.StringDtype(),
    "source": pd.StringDtype(),
    "category_within_source": pd.StringDtype(),
    "source_domain": pd.StringDtype(),
    "overall_sentiment_score": pd.Float64Dtype(),
    "overall_sentiment_label": pd.StringDtype(),
}

# Loop through each symbol, year, and month combination
n_symbols = len(symbols)

# initialize api call timestamp for later
api_call_ts = datetime.utcnow() 

for symbol in symbols:

    symbol_idx = symbols.index(symbol)

    progress_bar = tqdm(year_months, desc=f"Gathering News data for {symbol.ljust(4, ' ')} ({symbol_idx + 1}/{n_symbols})")
    
    n_no_results_for_symbol = 0

    for year, month in progress_bar:

        if n_no_results_for_symbol > NO_RESULT_LIMIT:
            print(f"Skipped {symbol} because no results were found for the last {NO_RESULT_LIMIT} months")
            break

        # Set the time_from parameter to the first day of the current month and year
        current_month = datetime(year, month, 1)
        time_from = current_month.strftime("%Y%m%dT0000")
        time_to = (current_month + relativedelta(months=1)).strftime("%Y%m%dT0000")

        # Check if the DataFrame for the current symbol, year, and month exists
        folder = Path(f"data/AlphaVantage/{symbol}")
        folder.mkdir(parents=True, exist_ok=True)

        filepath = folder / f"{year:04}-{month:02}.parquet"

        if os.path.exists(filepath):
            progress_bar.write(
                f"Skipped {symbol} {year:04}-{month:02}, file already exists"
            )
            progress_bar.update(1)
            continue

        # Format the API endpoint URL with the current symbol, year, month, and apikey
        endpoint = url.format(
            symbol=symbol, time_from=time_from, time_to=time_to, api_key=api_key
        )

        # time past since last api call
        pre_api_call_ts = datetime.utcnow()
        # sleep if needed to avoid rate limit 
        current_delay = (pre_api_call_ts - api_call_ts).total_seconds()
        time.sleep(max(0, min_delay_seconds - current_delay)) 

        # make api call
        response = session.get(endpoint)
        api_call_ts = datetime.utcnow() 

        # Convert the response to a JSON object
        data = response.json()

        if data.get("Information", "").startswith("No articles found"):
            progress_bar.write(
                f"Skipped {symbol} {year:04}-{month:02}, no articles found"
            )
            progress_bar.update(1)
            n_no_results_for_symbol += 1
            continue

        if 'feed' not in data:
            progress_bar.write(
                f"Skipped {symbol} {year:04}-{month:02}, unknown error"
            )
            continue

        # Convert the articles data to a Pandas DataFrame
        df = pd.json_normalize(data["feed"])

        if len(df) == 0:
            progress_bar.write(
                f"Skipped {symbol} {year:04}-{month:02}, no articles found"
            )
            progress_bar.update(1)
            n_no_results_for_symbol += 1
            continue

        # Convert the date column to a datetime
        df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S")
        df = df.astype(dtypes)

        # Write the DataFrame to a Parquet file
        df.to_parquet(filepath)

        # Update the progress bar and print a message indicating that the DataFrame has been saved
        progress_bar.write(f"Saved {symbol} {year:04}-{month:02}")
