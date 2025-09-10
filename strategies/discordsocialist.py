# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter

# bastardized SMA/EMA and v5 CBHAC by @smarm
# author @tirail and @iterativ for the original code and signals all credit to them 

class discordsocialist(IStrategy):
    INTERFACE_VERSION = 2
    
    buy_params = {
        "base_nb_candles_buy": 23,
        "buy_trigger": ta.SMA,
        "low_offset": 0.966,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 15,
        "high_offset": 0.867,
        "sell_trigger": ta.SMA,
    }

    # Stoploss:
    stoploss = -0.218
    timeframe = '5m'
    informative_timeframe = '1h'
    use_sell_signal = True
    sell_profit_only = False
    process_only_new_candles = True

    use_custom_stoploss = True
    startup_candle_count = 200

    # ROI table:
    minimal_roi = {
        "0": 0.232,
        "20": 0.05,
        "66": 0.025,
        "176": 0
    }

    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy')
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell')
    low_offset = DecimalParameter(0.8, 0.99, default=buy_params['low_offset'], space='buy')
    high_offset = DecimalParameter(0.8, 1.1, default=sell_params['high_offset'], space='sell')
    buy_trigger = CategoricalParameter([ta.SMA, ta.EMA], default=buy_params['buy_trigger'], space='buy')
    sell_trigger = CategoricalParameter([ta.SMA, ta.EMA], default=sell_params['sell_trigger'], space='sell')

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False

    process_only_new_candles = False
    startup_candle_count = 200

    plot_config = {
        'main_plot': {
            'ma_offset_buy': {'color': 'orange'},
            'ma_offset_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:

        if current_time - timedelta(minutes=40) > trade.open_date_utc and current_profit < -0.07:
            return -0.01

        return -0.99

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    @staticmethod
    def get_informative_indicators(dataframe: DataFrame, metadata: dict):

        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['go_long'] = (
               (dataframe['ema_fast'] > dataframe['ema_slow'])
       ).astype('int') * 2

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        if not self.dp:
            return dataframe

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        informative = self.get_informative_indicators(informative.copy(), metadata)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                           ffill=True)
        # don't overwrite the base dataframe's HLCV information
        skip_columns = [(s + "_" + self.informative_timeframe) for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.rename(
            columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s,
            inplace=True)

        # ---------------------------------------------------------------------------------

        sma_offset = (1 - 0.04)
        sma_offset_pos = (1 + 0.012)
        base_nb_candles = 20

        dataframe['sma_30_offset'] = ta.SMA(dataframe, timeperiod=base_nb_candles) * sma_offset
        dataframe['sma_30_offset_pos'] = ta.SMA(dataframe, timeperiod=base_nb_candles) * sma_offset_pos
        
        #strat SMAOffsetv2
        bb_40 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['lower'] = bb_40['lower']
        dataframe['mid'] = bb_40['mid']
        dataframe['bbdelta'] = (bb_40['mid'] - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
  
        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ma_offset_buy'] = self.buy_trigger.value(dataframe, int(self.base_nb_candles_buy.value)) * self.low_offset.value

        dataframe.loc[
            (
                # original SMAoffsetv2
                    (dataframe['close'] < dataframe['ma_offset_buy']) &
                    (dataframe['volume'] > 0)
            )
            |
            (  # strategy BinHV45
                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * 0.008) &
                dataframe['closedelta'].gt(dataframe['close'] * 0.0175) &
                dataframe['tail'].lt(dataframe['bbdelta'] * 0.25) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift()) &
                (dataframe['volume'] > 0)
            )
            |
            #(  # strategy ClucMay72018
            #    (dataframe['close'] < dataframe['ema_slow']) &
            #    (dataframe['close'] < 0.985 * dataframe['bb_lowerband']) &
            #    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 20)) &
            #    (dataframe['volume'] > 0)
            #)
            #|
            (
                (dataframe['go_long'] > 0)
                &
                (dataframe['close'] < dataframe['sma_30_offset'])
                &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['go_long'] == 0)
                    |
                    (dataframe['close'] > dataframe['sma_30_offset_pos'])
                )
                &
                (dataframe['volume'] > 0)
            )
            |
            ( # Improves the profit slightly.
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe