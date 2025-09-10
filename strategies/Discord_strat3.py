

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class strat3(IStrategy):
    """
    strat3 - work in progress!
    author@: me
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.18054,
        "37": 0.10473,
        "90": 0.03969,
        "210": 0.005,
        "350": 0.001
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'

    startup_candle_count = 40
    
    # trailing stoploss
    stoploss = -0.35
    trailing_stop = False
    trailing_stop_positive = 0.10
    trailing_stop_positive_offset = 0.20
    trailing_only_offset_is_reached = False

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }


    """
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        ""
        if current_profit < 0.06:
            return -1
        desired_stoploss = current_profit / 1.3
        return max(min(desired_stoploss, 0.10), 0.05)
        ""
        return -1
    """
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)

        stochf = ta.STOCHF(dataframe)
        dataframe['fastd'] = stochf['fastd']
        dataframe['fastk'] = stochf['fastk']

        stochs = ta.STOCH(dataframe)
        dataframe['slowk'] = stochs['slowk']
        dataframe['slowd'] = stochs['slowd']

        #dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        #dataframe['ao'] = awesome_oscillator(dataframe)

        #macd = ta.MACD(dataframe)
        #dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']

        #hilbert = ta.HT_SINE(dataframe)
        #dataframe['htsine'] = hilbert['sine']
        #dataframe['htleadsine'] = hilbert['leadsine']

        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
    
        # MACD
        #macd = ta.MACD(dataframe)
        #dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)
        
        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=10, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['vol_mean'] = dataframe['volume'].rolling(720).mean()
        dataframe['vol_std'] = ta.STDDEV(dataframe['volume'], timeperiod=720)
        dataframe['volz'] = (dataframe['volume'] - dataframe['vol_mean'])/dataframe['vol_std']

        # Hammer: values [0, 100]
        #dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        #dataframe['cci'] = ta.CCI(dataframe)
        
        dataframe['avolprev20'] = dataframe['volume'].rolling(20).mean() - dataframe['volume'] * 0.05
        dataframe['avolprev5'] = dataframe['volume'].rolling(5).mean() - dataframe['volume'] * 0.2
        
        dataframe['buysignal'] = (
            (dataframe['close'] < (dataframe['ema10'] * 0.997)) &
            (dataframe['close'] < (dataframe['sma'] * 0.995)) &
            (dataframe['close'] < dataframe['sar'])  &
            (dataframe['fastd'] > dataframe['fastk']) &
            (dataframe['fisher_rsi_norma'] < 15) &
            (dataframe['volume'] > 0)
        )
        
        dataframe['sellsignal'] = (
            (dataframe['plus_di'] > dataframe['minus_di']) &
            (dataframe['close'] > (dataframe['sma'] * 1.03)) &
            (dataframe['fisher_rsi_norma'] > 80) &
            (dataframe['ema5'] > dataframe['sma']) &
            (dataframe['volz'] >= 2)
        )
        
        #dataframe['buysignalcount'] = dataframe['buysignal'].rolling(4).sum()
        #dataframe['sellsignalcount'] = dataframe['sellsignal'].rolling(2).sum()
       
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
        (
            (dataframe['buysignal'])
        ),
        'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (
                    (dataframe['sellsignal']) & False
                )
            ),
            'sell'] = 1
        return dataframe

    unfilledtimeout = {
        'buy': 60 * 25,
        'sell': 60 * 25
    }
    """
    def check_buy_timeout(self, pair: str, trade: 'Trade', order: dict, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date < datetime.utcnow() - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date < datetime.utcnow() - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date < datetime.utcnow() - timedelta(hours=24):
           return True
        return False


    def check_sell_timeout(self, pair: str, trade: 'Trade', order: dict, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date < datetime.utcnow() - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date < datetime.utcnow() - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date < datetime.utcnow() - timedelta(hours=24):
           return True
        return False
    """
    plot_config = {
        'main_plot': {
            # Configuration for main plot indicators.
            # By omitting color, a random color is selected.
            'sar': {},
            'sma': {},
            'ema5': {},
            'tema': {}
        },
        'subplots': {
            "vol": {
                'avolprev20':{},
                'avolprev5':{},
                'volume':{}
            },
        		"BS": {
                'buysignal': {'color':'green'},
                'buysignalcount': {'color': 'green'},
                'sellsignal': {'color':'red'},
                'sellsignalcount': {'color':'red'}		
        		},
            "DI": {
              'plus_di': {'color': 'green'},
              'minus_di': {'color': 'red'}
            },
            "NF RSI": {
              'fisher_rsi_norma': {'color': 'green'}
            },
            "FKD": {
              'fastk': {'color': 'red'},
              'fastd': {'color': '#CCCCCC'}
            },
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'}
            },
            # Additional subplot RSI
            "RSI": {
                'rsi': {'color': 'red'}
            }
        }
    }
