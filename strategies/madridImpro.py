
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class madridImpro(IStrategy):
    INTERFACE_VERSION: int = 3
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }
    can_short = True
    stoploss = -0.10
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        stoploss_levels = {
            0.3: 0.01,
            0.1: 0.015,
            0.06: 0.01,
            0.02: 0.05,
            0.01: 0.003
        }
        for level, stoploss_value in stoploss_levels.items():
            if current_profit > level:
                return stoploss_value
        return 0.15


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        indicators = ["ema_madrid", "ema5", "ema10", "ema15", "ema20", "ema25", "ema30", "ema35",
                      "ema40", "ema45", "ema50", "ema55", "ema60", "ema65", "ema70", "ema75",
                      "ema80", "ema85", "ema90", "ema95", "ema100", "ema200", "rsi"]

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowk_matype=0,
                         slowd_period=3, slowd_matype=0)
        dataframe["slowd"] = stoch["slowd"]
        dataframe["slowk"] = stoch["slowk"]

        macd = ta.MACD(dataframe, fastperiod=12, fastmatype=0, slowperiod=26,
                       slowmatype=0, signalperiod=9, signalmatype=0)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        for indicator in indicators:
            dataframe[indicator] = ta.EMA(dataframe, timeperiod=int(indicator[3:]))
        
        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (dataframe["rsi"] > 51) & (dataframe['close'] > dataframe['bb_middleband']),
            (dataframe["rsi"] < 51) & (dataframe['close'] < dataframe['bb_middleband'])
        ]

        dataframe['enter_long'] = reduce(lambda x, y: x & y, conditions)
        dataframe['enter_short'] = reduce(lambda x, y: x & y, [~condition for condition in conditions])

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_long_exit = qtpylib.crossed_above(dataframe['ema10'], dataframe['ema100'])
        conditions_short_exit = qtpylib.crossed_below(dataframe['ema10'], dataframe['ema100'])

        dataframe['exit_long'] = conditions_long_exit
        dataframe['exit_short'] = conditions_short_exit

        return dataframe
