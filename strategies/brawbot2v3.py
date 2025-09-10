from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from datetime import datetime
import numpy as np
from freqtrade.persistence import Trade


class brawbot2(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.002  # NOTE: Good value (Win% ~70%), many trades
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_trend_indicator": "trend_close_2h",
    }

    # ROI table:
    minimal_roi = {
        "0": 0.059,
        "10": 0.037,
        "30": 0.024,
        "50": 0.015,
        "70": 0.009,
        "114": 0
    }

    # Stoploss:
    stoploss = -0.275

    # Optimal timeframe for the strategy
    timeframe = '5m'

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.006
    trailing_only_offset_is_reached = True


    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    plot_config = {
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(255,76,46,0.2)',
            },
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'}
        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            }
        }
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Initialize profit tracking variables
        self.daily_profit = 0
        self.monthly_profit = 0
        self.current_day = None

    def update_daily_and_monthly_profit(self):
        """
        Update the daily and monthly profit based on completed trades in a dry run.
        """
        # Get today's date and the current month
        today = datetime.now().date()
        current_month = today.strftime("%Y-%m")

        # Query all closed trades
        trades = Trade.get_trades()  # Fetch all trades from the database

        # Calculate daily profit (for trades closed today)
        self.daily_profit = sum(
            trade.close_profit for trade in trades if trade.close_date and trade.close_date.date() == today
        )

        # Calculate monthly profit (for trades closed this month)
        self.monthly_profit = sum(
            trade.close_profit for trade in trades if trade.close_date and trade.close_date.strftime("%Y-%m") == current_month
        )
        
        

    def should_allow_new_trades(self) -> bool:
        """
        Returns True if new trades are allowed based on daily/monthly profit rules.
        """
        # Update profit metrics
        self.update_daily_and_monthly_profit()

        # Allow unlimited trades if the monthly profit is negative
        if self.monthly_profit < 0:
            return True

        # Allow trades if the daily profit target of 2% has not been reached
        if self.daily_profit < 0.03:  # Daily profit is in percentage
            return True

        # Otherwise, do not allow new trades
        return False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=96)

        dataframe['trend_open_5m'] = dataframe['open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=96)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        # Add other indicators as needed

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Base condition to disable new buys
        if not self.should_allow_new_trades():
            dataframe['buy'] = 0
            return dataframe

        # Example buy conditions
        conditions = []
        conditions.append(dataframe['trend_close_5m'] > dataframe['trend_close_15m'])

        # Apply conditions to the 'buy' column
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
