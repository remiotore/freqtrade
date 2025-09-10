



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class FrayLIVEBTC15m(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
       "0":0.0136,
       "114":0.06,
       "192":0.021,
       "510":0.01
    }


    stoploss = -0.346

    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.039
    trailing_stop_positive_offset = 0.048  # Disabled / not configured



    buy_frsi = DecimalParameter(-0.5, 1, decimals = 3, default = 0.25, space="buy")

    buy_dip_frsi = DecimalParameter(-0.9, 0.2, decimals = 2, default = -0.7, space="buy")
    frsi_pct = DecimalParameter(0, 1, decimals = 4, default = 0.6, space="buy") #use pct rate to calc percentage of rsi rising against previous candles
    ema_pct = DecimalParameter(0, 0.1, decimals = 4, default = 0.08, space="buy")  #percentages of Difference between EMA7 against EMA7-TEMA
    macdn_buy = DecimalParameter(0, 0.8, decimals = 2, default = 0.09, space="buy")

    sell_frsi = DecimalParameter(-1, 1, decimals=2, default=-0.84, space="sell") #Main F-RSI
    macd_diff = DecimalParameter(0, 0.01, decimals=4, default=0.0047, space='sell') #Distance between MACD and MACD SIGNAL
    macdn_sell = DecimalParameter(0.2, 0.99, decimals=2, default= 0.81, space="sell") #MACD signal position near middle line

    cooldown_lookback = IntParameter(2, 90, default=3, space="protection", optimize=True)
    stop_duration = IntParameter(12, 100, default=3, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    @property
    def protections(self):
            return [
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self.cooldown_lookback.value
                },
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self.cooldown_lookback.value,
                    "trade_limit": 5,
                    "stop_duration_candles": self.stop_duration.value,
                    "max_allowed_drawdown": 0.9
                },
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 20,
                    "trade_limit": 3,
                    "stop_duration_candles": 4,
                    "only_per_pair": False
                },
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": 24,
                    "trade_limit": 2,
                    "stop_duration_candles": 4,
                    "required_profit": 0.01
                }
                
            ]

    timeframe = '15m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 40

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'blue'},
            'ema7':{'color': 'red'},
            'ema12':{'color': 'yellow'}
        },
        'subplots': {
            "MACD": {
                'macdn': {'color': 'blue'},
                'macdnsig': {'color': 'orange'}

            },
            
            "FISHERS RSI":{
                'frsi':{'color':'green'},
            }
        }
    }











    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """





        
        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['frsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

















        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']


        macdn = dataframe['macd']
        macdmin = (dataframe['macd'].min())
        macdmax = (dataframe['macd'].max())
        
        dataframe['macdn'] = (macdn - macdmin) / (macdmax - macdmin)
        
        macdnsig = dataframe['macdsignal']
        macdnsigmax = (dataframe['macdsignal'].max())
        macdnsigmin = (dataframe['macdsignal'].min())
        dataframe['macdnsig'] = (macdnsig - macdnsigmin) / (macdnsigmax - macdnsigmin)
































        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)

        dataframe['ema12'] = ta.EMA(dataframe, timeperiod=14)


        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=7)

















































        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][1]
                dataframe['best_ask'] = ob['asks'][0][0]
        

        return dataframe












    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        frsi_last3 = dataframe['frsi'].tail(3)
        frsi3rdlast, frsi2ndlast, frsilast = frsi_last3
        last_ema7 = dataframe['ema7']
        last_tema = dataframe['tema']
        macdn_buy_low = float(self.macdn_buy.value)       
        dataframe.loc[


            (
                (qtpylib.crossed_above(dataframe['frsi'], self.buy_frsi.value)) &

                (dataframe['macdnsig'] > dataframe['macdnsig'].shift(1)) &
                (dataframe['macdnsig'] < dataframe['macdn']) &

                (dataframe['sar'] <= dataframe['tema']) &
                (dataframe['volume'] > 0)
             ),
             'buy'] = 1
        return dataframe

        dataframe.loc[

            (
                (last_tema < last_ema7 ) &
                (last_ema7 < dataframe['ema12']) &
                (qtpylib.crossed_below(dataframe['frsi'], self.buy_dip_frsi.value)) &
                (abs(frsilast - frsi2ndlast) < abs(frsi2ndlast - frsi3rdlast) < self.frsi_pct.value ) &
                (((last_ema7 - last_tema) / last_tema ) >= self.ema_pct.value ) & #Guard against False Bottom
                (dataframe['macdn'] <= macdn_buy_low ) &
		(dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        HUEHUEHUE
        """
        last_macd = dataframe['macd']
        last_macdsig = dataframe['macdsignal']
        dataframe.loc[
             (
                (dataframe['tema'] > dataframe['ema7']) &
                (dataframe['tema'] < dataframe['tema'].shift(1)) &

                (qtpylib.crossed_above(dataframe['frsi'], self.sell_frsi.value )) &

                ((( last_macd - last_macdsig ) / last_macdsig ) <= self.macd_diff.value ) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            
             ),
             'sell'] = 1
        return dataframe
        dataframe.loc[
             (
                (((last_ema7 - last_tema) / last_tema ) >= self.ema_pct.value ) &
                (dataframe['macdn'] <= self.macdn_sell.value) &
                (dataframe['volume'] > 0 )
             ),
             'sell'] = 1
        return dataframe
