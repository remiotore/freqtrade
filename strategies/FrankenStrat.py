import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, stoploss_from_open, DecimalParameter, IntParameter, CategoricalParameter

from functools import reduce












































def SSLChannels(dataframe, length = 7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

def ewo(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

class FrankenStrat(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.029,          # I feel lucky!
        "10": 0.021,
        "30": 0.01,
        "40": 0.005,
    }

    base_nb_candles_buy = IntParameter(5, 80, default=12, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=31, space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=0.975, space='buy', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=1.0, space='sell', optimize=True)

    tema_low_offset = DecimalParameter(0.9, 0.99, default=0.95, space='buy', optimize=True)

    fast_ewo = 50
    slow_ewo = 200
    
    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h'

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = False

    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    use_custom_stoploss = True

    process_only_new_candles = False

    startup_candle_count: int = 200

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        if (current_profit < 0) & (current_time - timedelta(minutes=240) > trade.open_date_utc):
            return 0.01
        return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."

        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        informative_1h['ssl_down'] = ssl_down_1h
        informative_1h['ssl_up'] = ssl_up_1h

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for length in set(list(self.base_nb_candles_buy.range) + list(self.base_nb_candles_sell.range)):
            dataframe[f'ema_{length}'] = ta.EMA(dataframe, timeperiod=length)
        
        dataframe['ewo'] = ewo(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)        
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe['tema'] = ta.TEMA(dataframe, length=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ema_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                
                (dataframe['ewo'] > -1) &
                
                (dataframe['volume'] > 0)
            )
            |
            (
                (dataframe['close'] < (dataframe['tema'] * self.tema_low_offset.value)) &
                
                (dataframe['ewo'] > -1) &
                
                (dataframe['volume'] > 0)
            )
            |

            (  # strategy ClucMay72018
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < 0.99 * dataframe['bb_lowerband']) &                           # Guard is on, candle should dig not so hard (0,99)
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * 0.4) &   # Try to exclude pumping

                (dataframe['volume'] > 0)
            )
            |

            (  # strategy ClucMay72018 
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < 0.975 * dataframe['bb_lowerband']) &                          # Guard is off, candle should dig hard (0,975) 
                (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &                         # Don't buy if someone drop the market.
                (dataframe['rsi_1h'] < 15) &                                                        # Buy only at dip
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * 0.4) &   # Try to exclude pumping
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |   

            (  # strategy MACD Low buy 
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &                         # Don't buy if someone drop the market.
                (dataframe['close'] < (dataframe['bb_lowerband'])) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * 0.4) &   # Try to exclude pumping
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |

            (  # strategy MACD Low buy 
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.03)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &                 # Don't buy if someone drop the market.
                (dataframe['close'] < (dataframe['bb_lowerband'])) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |
            (
                (dataframe['close'] < dataframe['sma_5']) &
                (dataframe['ssl_up_1h'] > dataframe['ssl_down_1h']) &
                (dataframe['ema_slow'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['rsi'] < dataframe['rsi_1h'] - 43.276) &
                (dataframe['volume'] > 0)
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(
            (
                (dataframe['close'] > dataframe['bb_middleband'] * 1.01) &                  # Don't be gready, sell fast
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
        )
        
        conditions.append(
            (
                (dataframe['close'] > (dataframe[f'ema_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
                &
                (dataframe['ewo'] < 20)
                &
                (dataframe['volume'] > 0)
            ) 
        )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1
            
        return dataframe