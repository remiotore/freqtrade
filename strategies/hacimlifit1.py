



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from technical.indicators import fibma,vwma
from functools import reduce
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import BooleanParameter,DecimalParameter,IStrategy,IntParameter


import talib.abstract as ta
from typing import List

def get_whale_volume_threshold(dataframe:DataFrame,mean_multiplier=3,volmean=200):
    data = dataframe.copy()
    data["mean_volume"] = data['volume'].rolling(volmean).mean()
    data["std_volume"] = data['volume'].rolling(volmean).std()

    data["std_threshold"] = round(data["mean_volume"] + mean_multiplier * data["std_volume"],2)
    return data["std_threshold"]

class hacimlifit1(IStrategy):


    INTERFACE_VERSION = 3

    can_short: bool = False

    class HyperOpt:

        def stoploss_space():
            return [SKDecimal(-0.5, -0.15, decimals=2, name='stoploss')]
        def trailing_space() -> List[Dimension]:

            return [

                Categorical([True], name='trailing_stop'),

                SKDecimal(0.01, 0.35, decimals=3, name='trailing_stop_positive'),





                SKDecimal(0.05, 0.3, decimals=3, name='trailing_stop_positive_offset_p1'),

                Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]

        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(-1, 8, name='max_open_trades'),
            ]


    minimal_roi = {


        "0": 10,
    }


    stoploss = -0.46

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.152
    trailing_only_offset_is_reached = True

    timeframe = "5m"

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    buy_dfma = DecimalParameter(1.01, 1.06, default=1.059, space="buy", optimize=True, load=True)

    upwneed = BooleanParameter(default=True, space="buy", optimize=True)
    buy_upw = DecimalParameter(1.001, 1.08, default=1.044, space="buy", optimize=True, load=True)  

    buy_cs = DecimalParameter(1.02, 1.05, default=1.03, space="buy", optimize=True, load=True)

    fibmaneeed = BooleanParameter(default=True, space="buy", optimize=True)

    buy_adx = IntParameter(15, 60, default=20, space="buy", optimize=True, load=True)

    emaneed = BooleanParameter(default=True, space="buy", optimize=True)

    vwmaneed = BooleanParameter(default=True, space="buy", optimize=True)

    badtime = BooleanParameter(default=True, space="buy", optimize=True)
    houra = IntParameter(0, 8, default=4, space="buy", optimize=True, load=True)
    hourb = IntParameter(9, 16, default=11, space="buy", optimize=True, load=True)
    hourc = IntParameter(17, 24, default=18, space="buy", optimize=True, load=True)

    lastvlneed = BooleanParameter(default=True, space="buy", optimize=True)
    buy_lastvl = IntParameter(2, 10, default=2, space="buy", optimize=True, load=True)
    last_lowest = IntParameter(10, 100, default=24, space="sell", optimize=True, load=True)

    cooldown_lookback = IntParameter(2, 48, default=15, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=135, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    stop_max_allowed_dd = DecimalParameter(0.05, 0.5, default=0.108, space="protection", optimize=True)
    use_MaxDrawdown_protection = BooleanParameter(default=True, space="protection", optimize=True)
    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 4,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })
        if self.use_MaxDrawdown_protection.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 20,
                "stop_duration_candles": self.stop_duration.value,
                "max_allowed_drawdown": self.stop_max_allowed_dd.value
            })

        return prot

    startup_candle_count: int = 200

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "fibma": {"color": "white"},
            "ema200": {"color": "purple"},
            "vwma200": {"color": "green"},
        },
        "subplots": {
            "ADX": {
                "adx": {"color": "yellow"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def leverage(self, pair: str,  current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
 
        return 10.0        

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['hour'] = dataframe['date'].dt.hour
        dataframe['minute'] = dataframe['date'].dt.minute
        dataframe["whale_volume_threshold"] = get_whale_volume_threshold(dataframe,volmean=200,mean_multiplier=5)
        dataframe['volinc'] = 0
        dataframe.loc[dataframe['volume'] > dataframe["whale_volume_threshold"], 'volinc'] = 1
        dataframe["volcheck"] = dataframe['volume']> dataframe["whale_volume_threshold"]
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['fibma'] = fibma(dataframe, MAtype=1, src=1)
        dataframe['obv_slope'] = dataframe['obv'] - dataframe['obv'].shift(1)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe["close"],14)
        dataframe['ema200'] = ta.EMA(dataframe["close"],200)
        dataframe["vwma200"] = vwma(dataframe,"close",200)

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        longconditions = []

        longconditions.append(dataframe["close"] < dataframe["open"]*self.buy_cs.value)        
        longconditions.append(dataframe["close"] < dataframe["fibma"]*self.buy_dfma.value)
        if self.upwneed.value:
            longconditions.append(dataframe["close"] * self.buy_upw.value > dataframe["high"])
        if self.fibmaneeed.value:
            longconditions.append(dataframe["fibma"] > dataframe["fibma"].shift(1))

        longconditions.append(dataframe["close"] > dataframe["open"])

        longconditions.append(dataframe['adx'] > self.buy_adx.value)
        longconditions.append(dataframe['obv_slope'] > 0)
        if self.emaneed.value:
            longconditions.append(dataframe['close'] > dataframe['ema200'] )
        if self.vwmaneed.value:
            longconditions.append(dataframe['close'] > dataframe['vwma200'])
        if self.badtime.value:
            longconditions.append(~((dataframe['hour'] >= self.houra.value) & (dataframe['hour'] < self.houra.value+2)))
            longconditions.append(~((dataframe['hour'] >= self.hourb.value) & (dataframe['hour'] < self.hourb.value+2)))
            longconditions.append(~((dataframe['hour'] >= self.hourc.value) & (dataframe['hour'] < (self.hourc.value+2)%24)))

        longconditions.append(dataframe['volcheck'] == True )
        if self.lastvlneed.value:
            longconditions.append(dataframe['volinc'].rolling(window=24 ).sum() < self.buy_lastvl.value)

        longconditions.append(dataframe['volume'] > 0)
        if longconditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, longconditions),
                'enter_long'] = 1

        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (   
                    (dataframe['close'] < dataframe['low'].rolling(self.last_lowest.value).min().shift(1))  
                )
            ),

            'exit_long'] =1
        return dataframe
