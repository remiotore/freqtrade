# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------


class bbandrsi(IStrategy):
    """
    author@: Gert Wohlgemuth
    converted from:
    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs
    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0":  0.03279,
        "259": 0.02964,
        "536" : 0.02467,
        "818": 0.02326,
        "965": 0.01951,
        "1230": 0.01492,
        "1279" : 0.01502,
        "1448": 0.00945,
        "1525" : 0.00698,
        "1616": 0.00319,
        "1897" : 0
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    plot_config = {
        'main_plot': {
            'bb_lowerband': {},
            'bb_middleband': {},
            'bb_upperband': {}
        },
        'subplots': {
            "MFI": {
                'mfi': {'color': 'blue'}
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=10)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
	# EMA
        dataframe['ema1'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema2'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema4'] = ta.EMA(dataframe, timeperiod=25)
        #CHOPPINESS INDEX
        dataframe['chop']= qtpylib.chopiness(dataframe, window=14)
        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['mfi'] < 20) &
                    (dataframe['rsi'] < 37) &
                    (dataframe['ema2'] < dataframe['ema1']) &
                    (dataframe['close'] < dataframe['bb_lowerband'])
            ),
            ['buy', 'buy_tag']] = (1, 'bbmfi')

        dataframe.loc[
            (
                    (dataframe['close'].shift(2) < dataframe['close'].shift(1)) &
                    (dataframe['close'].shift(1) < dataframe['close']) &
                    (dataframe['chop'].shift(1) > dataframe['chop']) &
                    (dataframe['rsi'] > 55) &
                    (dataframe['rsi'] < 65) &
                    (dataframe['ema3'].shift(1) < dataframe['ema3'])
            ),
            ['buy', 'buy_tag']] = (1, 'teste')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 70)

            ),
            'sell'] = 1
        return dataframe