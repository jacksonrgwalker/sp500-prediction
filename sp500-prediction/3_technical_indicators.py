import pandas as pd
from pathlib import Path
from stockstats import wrap


def calculate_technical_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    df = wrap(ohlc)

    df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    df["cmf"] = df["clv"] * df["volume"]
    df["cmf"] = (
        df["cmf"].rolling(window=20).sum() / df["volume"].rolling(window=20).sum()
    )
    df["atr_percent"] = df["atr"] / df["close"]

    tech_indicator_cols = [
        "adx",
        "trix",
        "adxr",
        "cci",
        "macd",
        "macdh",
        "rsi_14",
        "kdjk",
        "wr_14",
        "atr_percent",
        "atr",
        "cmf",
    ]
    return df[tech_indicator_cols]


ohlc_save_path = Path("data/ohlc.parquet")
ohlc = pd.read_parquet(ohlc_save_path)

technical = ohlc.groupby("symbol").apply(calculate_technical_indicators)
technical = technical.reset_index(level=0, drop=True)
technical.sort_index(inplace=True)

technical_save_path = Path("data/technical_indicators.parquet")
technical.to_parquet(technical_save_path)
