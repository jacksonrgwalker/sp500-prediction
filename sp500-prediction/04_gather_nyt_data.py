import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import Retry
from tqdm import tqdm
from datetime import datetime

assert load_dotenv(), "Failed to load .env file"

dtypes = {
    "abstract": pd.StringDtype(),
    "web_url": pd.StringDtype(),
    "snippet": pd.StringDtype(),
    "lead_paragraph": pd.StringDtype(),
    "print_section": pd.StringDtype(),
    "print_page": pd.StringDtype(),
    "source": pd.StringDtype(),
    "document_type": pd.StringDtype(),
    "news_desk": pd.StringDtype(),
    "section_name": pd.StringDtype(),
    "type_of_material": pd.StringDtype(),
    "_id": pd.StringDtype(),
    "word_count": pd.Int64Dtype(),
    "uri": pd.StringDtype(),
    "headline.main": pd.StringDtype(),
    "headline.kicker": pd.StringDtype(),
    "headline.content_kicker": pd.StringDtype(),
    "headline.print_headline": pd.StringDtype(),
    "headline.name": pd.StringDtype(),
    "headline.seo": pd.StringDtype(),
    "headline.sub": pd.StringDtype(),
    "byline.original": pd.StringDtype(),
    "byline.organization": pd.StringDtype(),
    "subsection_name": pd.StringDtype(),
}

# Create a list of years and months
years = list(range(2000, 2024))
months = list(range(1, 13))
year_months = [(year, month) for year in years for month in months]

# Define the retry strategy in case of a failed request
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)

# set max backoff to 10 seconds
retry_strategy.DEFAULT_BACKOFF_MAX = 10

# Create a session with the retry adapter
session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry_strategy))

# save API key as NYT_API_KEY (see https://developer.nytimes.com/apis)
api_key = os.environ["NYT_API_KEY"]

# Set the API endpoint URL
url = (
    f"https://api.nytimes.com/svc/archive/v1/{{year}}/{{month}}.json?api-key={api_key}"
)


# Loop through each year and month combination
progress_bar = tqdm(year_months, desc="Gathering NYT data")
for year, month in progress_bar:

    # Show current year and month in the progress bar
    progress_bar.set_postfix_str(f"{year:04}-{month:02}")

    # if at or after current year and month, skip
    if datetime(year, month, 1) >= datetime.now():
        progress_bar.write(f"Skipped {year}-{month}, current month or later")
        continue
    
    # Check if the Parquet file for the current year and month exists
    folder = Path(f"data/NYT/{year:04}")

    # Create the year folder if it doesn't exist
    folder.mkdir(parents=True, exist_ok=True)

    # Create the filepath for the current year and month
    filepath = folder / f"NYT-{year:04}-{month:02}.parquet"

    # Skip the current year and month if the Parquet file already exists
    if os.path.exists(filepath):
        progress_bar.write(f"Skipped {year}-{month}, file already exists")
        continue

    # Format the API endpoint URL with the current year and month
    endpoint = url.format(year=year, month=month)

    # Send a GET request to the API endpoint URL with the session
    response = session.get(endpoint)

    # Convert the response to a JSON object
    data = response.json()

    # Extract the articles from the JSON object
    articles = data["response"]["docs"]

    # Convert the articles to a Pandas DataFrame
    df = pd.json_normalize(articles)

    # Convert the pub_date column to a datetime
    df["pub_date"] = pd.to_datetime(df["pub_date"], format="%Y-%m-%dT%H:%M:%S%z")

    # Drop the multimedia column
    df.drop(columns=["multimedia"], inplace=True)

    # Convert the columns to the correct data types
    df = df.astype(dtypes)

    # Write the DataFrame to a Parquet file
    df.to_parquet(filepath)

    progress_bar.write(f"Saved {year:04}-{month:02} to {filepath}")

print("Done downloading NYT data")
