import pandas as pd

#df = pd.read_parquet('/Users/ray/Desktop/Claude Code/ibkr-signal-assistant/new/data/training_universe/parquet/clean/ohlcv/0003_HK.parquet')
df = pd.read_parquet('/Users/ray/Desktop/Claude Code/ibkr-signal-assistant/new/data/training_universe/parquet/raw/yahoo/ohlcv/0003_HK.parquet')

print(df.tail(100))