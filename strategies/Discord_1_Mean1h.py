# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
class Mean1h(IStrategy):
    ticker_interval = '1h'
    startup_candle_count: int = (24*60)/15+20
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    minimal_roi = {
        "0": 0.1
    }
    # Stoploss:
    stoploss = -0.075
    # Optimal timeframe for the strategy
    trailing_only_offset_is_reached = True
    trailing_stop = True
    trailing_stop_positive = 0.07
    trailing_stop_positive_offset = 0.09

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not {'buy', 'sell'}.issubset(dataframe.columns):
            dataframe.loc[:, 'buy'] = 0
            dataframe.loc[:, 'sell'] = 0
        dataframe['typical'] = qtpylib.typical_price(dataframe)
        dataframe['typical_sma'] = qtpylib.sma(dataframe['typical'],window=10)
        min = dataframe['typical'].shift(20).rolling(int(12 * 60 / 15)).min()
        max = dataframe['typical'].shift(20).rolling(int(12 * 60 / 15)).max()
        dataframe['daily_mean'] = (max+min)/2

        # EMA - Simple Moving Average
        dataframe['ma300'] = ta.SMA(dataframe, timeperiod=300)

        # EMA - Exponential Moving Average
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)


        return dataframe
    
    def informative_pairs(self):
        informative_pairs = []
        return informative_pairs
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['ema50'] > dataframe['ema100']) &
                (dataframe['ema20'] > dataframe['ema50']) &
                (qtpylib.crossed_below(dataframe['daily_mean'],dataframe['typical_sma']))
                # (dataframe["volume"] > dataframe["volume"].rolling(200).mean() * 4)

            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['daily_mean'],dataframe['typical_sma']))
            ),
            'sell'] = 1

        return dataframe