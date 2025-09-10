# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.state import RunMode
import logging
logger = logging.getLogger(__name__)

class TakeprofitThenTrail(IStrategy):
    """
    Until Takeprofit use inital stoploss
    Then use trailing stoploss
    """

    custom_info = {
            'initial_stoploss_modifier': 1,
            'risk_reward_ratio': 1.5,
            'trailing_stoploss_modifier': 2
            }
    use_custom_stoploss = True
    stoploss = -0.9
    sell_profit_only = True
    sell_profit_offset = 1 # 100%, get's set dynamically in trail

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        result = 1
        custom_info_pair = self.custom_info[pair]
        if custom_info_pair is not None:
            # using current_time/open_date directly will only work in backtesting/hyperopt.
            # in live / dry-run, we have to search for nearest row before
            tz = custom_info_pair.index.tz
            open_date = trade.open_date_utc if hasattr(
                trade, 'open_date_utc') else trade.open_date.replace(tzinfo=custom_info_pair.index.tz)
            open_date_mask = custom_info_pair.index.unique().get_loc(open_date, method='ffill')
            open_df = custom_info_pair.iloc[open_date_mask]

            # trade might be open too long for us to find opening candle
            if(open_df is None or len(open_df) == 0):
                return -1 # won't update current stoploss

            initial_sl_abs = open_df['initial_stoploss_rate']
            # calculate initial stoploss at open_date
            # use initial_sl
            initial_sl = initial_sl_abs/current_rate-1

            # calculate take profit treshold
            # by using the initial risk and multiplying it
            risk_distance = trade.open_rate-initial_sl_abs
            reward_distance = risk_distance*self.custom_info['risk_reward_ratio']
            # take_profit tries to lock in profit once price gets over
            # risk/reward ratio treshold
            take_profit_price_abs = trade.open_rate+reward_distance
            # take_profit gets triggerd at this profit
            take_profit_pct = take_profit_price_abs/current_rate-1

            # quick exit if still under takeprofit, use intial sl
            if (current_profit < take_profit_pct):
                return initial_sl

            result = initial_sl

            trailing_sl_abs = None
            # calculate current trailing stoploss
            if self.dp:
                # backtesting/hyperopt
                if self.dp.runmode.value in ('backtest', 'hyperopt'):
                    trailing_sl_abs = custom_info_pair.loc[current_time]['trailing_stoploss_rate']
                # for live, dry-run, storing the dataframe is not really necessary,
                # it's available from get_analyzed_dataframe()
                else:
                    # so we need to get analyzed_dataframe from dp
                    dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                             timeframe=self.timeframe)
                    # only use .iat[-1] in live mode, otherwise you will look into the future
                    # see: https://www.freqtrade.io/en/latest/strategy-customization/#common-mistakes-when-developing-strategies
                    trailing_sl_abs = dataframe['trailing_stoploss_rate'].iat[-1]

            if (trailing_sl_abs is not None and trailing_sl_abs < take_profit_price_abs):

                # calculate new_stoploss relative to current_rate
                # turn into relative negative offset required by `custom_stoploss` return implementation
                new_sl_relative = trailing_sl_abs/current_rate-1
                result = new_sl_relative

        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe['min'] = dataframe['low'].rolling(48).min()
        # dataframe['atr'] = ta.ATR(dataframe)
        # dataframe['initial_stoploss_rate'] = dataframe['min']-(dataframe['atr']*self.custom_info['initial_stoploss_modifier'])
        # dataframe['trailing_stoploss_rate'] = dataframe['low']-(dataframe['atr']*self.custom_info['trailing_stoploss_modifier'])
        self.custom_info[metadata['pair']] = dataframe[[
            'date', 'initial_stoploss_rate', 'trailing_stoploss_rate']].copy().set_index('date')

        # all "normal" indicators:
        # e.g.
        # dataframe['rsi'] = ta.RSI(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder Strategy: buys when SAR is smaller then candle before
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        # Deactivated buy signal to allow the strategy to work correctly
        dataframe.loc[:, 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder Strategy: does nothing
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        # Deactivated sell signal to allow the strategy to work correctly
        dataframe.loc[:, 'sell'] = 1
        return dataframe
