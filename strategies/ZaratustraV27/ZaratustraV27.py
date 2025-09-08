# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy, Trade, informative, IntParameter, CategoricalParameter
import talib.abstract as ta
from technical import qtpylib


class ZaratustraV27(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    use_exit_signal = False
    exit_profit_only = True
    position_adjustment_enable = True

    # ROI table:
    minimal_roi = {}

    # Base Stoploss:
    stoploss = -0.99

    # Trailing Stop Settings:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Max Open Trades
    max_open_trades = 1

    # Optional Risk/Trade Management parameters
    max_dca_orders = 3
    max_exit_position_adjustment = 2

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 10

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands for trend context
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe[['BBL', 'BBM', 'BBU']] = bollinger[['lower', 'mid', 'upper']]

        # RSI: Momentum and overbought/oversold indicator
        dataframe['RSI'] = ta.RSI(dataframe)

        # ADX and DI components: measure trend strength and direction
        dataframe['ADX'] = ta.ADX(dataframe)
        dataframe['PDI'] = ta.PLUS_DI(dataframe)
        dataframe['MDI'] = ta.MINUS_DI(dataframe)

        # TSF: Time Series Forecast as a trend direction indicator
        dataframe['TSF'] = ta.TSF(dataframe)

        # ATR: Average True Range to capture volatility (useful for adaptive stoploss)
        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bullish entry: Price above the mid Bollinger band and ADX confirms trend strength
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['BBM']) &
                (dataframe['ADX'] > 20) &
                (dataframe['PDI'] > dataframe['MDI'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish trend')

        # Bearish entry: Price below the mid Bollinger band and ADX confirms trend strength
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['BBM']) &
                (dataframe['ADX'] > 20) &
                (dataframe['MDI'] > dataframe['PDI'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish trend')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Custom exit logic is handled via custom_exit function.
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: float, max_stake: float, **kwargs) -> float:
        """
        Dynamically adjust trade position:
          - Scale into losing positions if RSI is favorable and TSF confirms trend.
          - Scale out of winning positions when RSI indicates exhaustion.
        """
        # Get the most recent candle data for the pair
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        rsi = last_candle['RSI']
        tsf = last_candle['TSF']
        adx = last_candle['ADX']

        filled_entries = trade.select_filled_orders('enter_short' if trade.is_short else 'enter_long')
        filled_exits = trade.select_filled_orders('exit_short' if trade.is_short else 'exit_long')

        count_entries = len(filled_entries)
        count_exits = len(filled_exits)

        # ------ Scale Into the Position (DCA) ------
        # Conditions: trade is losing, market oversold for longs or overbought for shorts,
        # the price respects the TSF line, and the overall trend is strong (ADX > 20)
        if current_profit < 0 and count_entries < self.max_dca_orders and adx > 20:
            if not trade.is_short and rsi < 40 and current_rate > tsf:
                stake_amount = min(trade.stake_amount * 1.5, max_stake)
                return stake_amount
            elif trade.is_short and rsi > 60 and current_rate < tsf:
                stake_amount = min(trade.stake_amount * 1.5, max_stake)
                return stake_amount

        # ------ Scale Out of the Position ------
        # Conditions: trade is profitable; RSI indicates potential overextension;
        # and only scale out a limited number of times.
        if current_profit > 0.05 and count_exits < self.max_exit_position_adjustment:
            if not trade.is_short and rsi > 70:
                return -(trade.amount * 0.25)  # exit 25% of the position
            elif trade.is_short and rsi < 30:
                return -(trade.amount * 0.25)

        return None
