what is the best way to get the OHLCV for 90 days ? my strategy will be using the 1d timeframe as i will be calculating both the ema 10 and ema 90 to get a volatility score based on the highs/lows Avg 

im using a custom strategy code but it seems like i'm getting candles of 2022 and 2019 instead of a lookback of a 90 days prior

```python
class WEMA(IStrategy):
    INTERFACE_VERSION: int = 3
    EMA_SHORT_TERM = 10
    EMA_MEDIUM_TERM = 12
    EMA_LONG_TERM = 90
    VOLATILITY_THRESHOLD = 0.1  # 10% threshold

    minimal_roi = {"0": 0.1}
    stoploss = -0.05
    timeframe = '1d'


        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_close_short'] = ta.EMA(dataframe['close'], timeperiod=self.EMA_SHORT_TERM)
        dataframe['ema_close_medium'] = ta.EMA(dataframe['close'], timeperiod=self.EMA_MEDIUM_TERM)
        dataframe['ema_close_long'] = ta.EMA(dataframe['close'], timeperiod=self.EMA_LONG_TERM)

        dataframe['ema_high_short'] = ta.EMA(dataframe['high'], timeperiod=self.EMA_SHORT_TERM)
        dataframe['ema_high_medium'] = ta.EMA(dataframe['high'], timeperiod=self.EMA_MEDIUM_TERM)
        dataframe['ema_high_long'] = ta.EMA(dataframe['high'], timeperiod=self.EMA_LONG_TERM)

        dataframe['ema_low_short'] = ta.EMA(dataframe['low'], timeperiod=self.EMA_SHORT_TERM)
        dataframe['ema_low_medium'] = ta.EMA(dataframe['low'], timeperiod=self.EMA_MEDIUM_TERM)
        dataframe['ema_low_long'] = ta.EMA(dataframe['low'], timeperiod=self.EMA_LONG_TERM)
        ema_values_df = DataFrame()
        ema_low_short_nth_value = dataframe['ema_low_short'].iloc[self.EMA_SHORT_TERM - 1]
        ema_high_short_nth_value = dataframe['ema_high_short'].iloc[self.EMA_SHORT_TERM - 1]
        ema_low_long_nth_value = dataframe['ema_low_long'].iloc[self.EMA_LONG_TERM - 1]
        ema_high_long_nth_value = dataframe['ema_high_long'].iloc[self.EMA_LONG_TERM - 1]
        ema_values_df['Date'] = [dataframe['date'].iloc[-1]]
        ema_values_df['Volume'] = [dataframe['volume'].iloc[-1]]
        
        ema_values_df['Pair'] = [metadata['pair']]
        ema_values_df['ema_low_short'] = [ema_low_short_nth_value]
        ema_values_df['ema_high_short'] = [ema_high_short_nth_value]
        ema_values_df['ema_low_long'] = [ema_low_long_nth_value]
        ema_values_df['ema_high_long'] = [ema_high_long_nth_value]
        ema_values_df['TimeFrame'] = [self.timeframe]
        
        dataframe.to_csv('ohlcv.csv', index=False)
        # ema_values_df.to_csv('ema_values.csv', index=False)
        return dataframe`
```
any advices for this ? i'm receiving lots of errors when trying to export to csv too