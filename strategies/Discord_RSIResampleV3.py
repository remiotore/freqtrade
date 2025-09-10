# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

rsi_trigger = 45

class RSIResampleV3(IStrategy):
    # ROI table:
    minimal_roi = {
        "0": 1,
    }

    # Stoploss:
    stoploss = -1

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.5
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_sell_signal = True

    plot_config = {
        'main_plot': {
            'sma_5': {'color': 'orange'},
        },
        'subplots': {
            "Signals": {
                'rsi': {'color': 'blue'},
                'rsi_max': {'color': 'green'},
                'rsi_max_sma': {'color': 'red'},
                'rsi_min_sma': {'color': 'yellow'},
            },
        }
    }

    use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=600) > trade.open_date_utc:
            return -0.05
        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['rsi_max'] = ta.MAX(dataframe['rsi'], timeperiod = 12 * 6)
        dataframe['rsi_max_sma'] = ta.SMA(dataframe['rsi_max'], timeperiod = 12 * 24 * 14)

        dataframe['rsi_min'] = ta.MIN(dataframe['rsi'], timeperiod = 12 * 6)
        dataframe['rsi_min_sma'] = ta.SMA(dataframe['rsi_min'], timeperiod = 12 * 24 * 14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
				(dataframe['rsi_max'] < 85) & # dont buy after big pumps
				(dataframe['rsi_max'] < dataframe['rsi_max_sma']) &
				(dataframe['rsi'] < rsi_trigger) &
				(dataframe['rsi'] < dataframe['rsi_min_sma'] - 5) &
				(dataframe['close'] < dataframe['sma_5']) &
				(dataframe['volume'] > 0)
            ),
			'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
					(
                    	(dataframe['rsi'] > 65) &
                    	(dataframe['rsi'] >= dataframe['rsi_max_sma'])
					) |
					(
						(dataframe['rsi'].shift(1) > 65) &
						(dataframe['rsi'].shift(1) >= dataframe['rsi_max_sma'].shift(1))
					)
				)&
				(dataframe['volume'] < dataframe['volume'].shift(1)) & # wait till raise end
				(dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
