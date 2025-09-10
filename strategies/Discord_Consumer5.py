import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, merge_informative_pair, stoploss_from_open
from functools import reduce

## Missing Imports
from pandas import DataFrame

## DCA Imports
import math
import logging

logger = logging.getLogger(__name__)

class Consumer5(IStrategy):
    timeframe = '1m'
    sell_params = {
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.16,
        "pPF_1": 0.0125,
        "pPF_2": 0.040,
        "pSL_1": 0.010,
        "pSL_2": 0.020
    }

    minimal_roi = {
    "0": 100,
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.15   # use custom stoploss

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.028
    trailing_only_offset_is_reached = True
    #...
    process_only_new_candles = False # required for consumers
    use_custom_stoploss = True

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 1,
                "stop_duration_candles": 480,
                "required_profit": -0.10,
                "only_per_pair": False,
                "only_per_side": False
            }
        ]

#######
    _producers = ['Consumer']
    _producer_tfs = {
        'Consumer': '1m'
    }

    _columns_to_expect = {}

    for producer in _producers:
        _columns_to_expect[producer] = [
            f'enter_long_Cluc_{producer}',
            f'enter_long_NFIX_{producer}',
            f'enter_long_BB_RPB_{producer}',
            f'enter_long_Elliot_{producer}',
            f'enter_long_{producer}',
            f'enter_tag_{producer}',
        ]
#######
    # def confirm_trade_entry(
    #     self,
    #     pair: str,
    #     order_type: str,
    #     amount: float,
    #     rate: float,
    #     time_in_force: str,
    #     current_time,
    #     entry_tag,
    #     side: str,
    #     **kwargs,
    # ) -> bool:
    #
    #     df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = df.iloc[-1].squeeze()
    #     prev_last_candle = df.iloc[-2].squeeze()
    #
    #     # if (abs(last_candle['zscore_DI_values']) > 2):
    #     #     self.dp.send_msg(f"Aborting entry for pair {pair}. Outlier detected.")
    #     #     return False
    #
    #     # if (last_candle['DI_values'] > last_candle['DI_outliers']):
    #     #     self.dp.send_msg(f"Aborting entry for pair {pair}. Outlier detected.")
    #     #     return False
    #     # elif (abs(last_candle['zscore_DI_values']) > 2.5):
    #     #     self.dp.send_msg(f"Aborting entry for pair {pair}. Outlier detected.")
    #     #     return False
    #     # elif (last_candle['do_predict'] != 1):
    #     #     self.dp.send_msg(f"Aborting entry for pair {pair}. Outlier detected.")
    #     #     return False
    #
    #     # else:
    #     if side == "long":
    #         # if last_candle['vwap_target_smooth_supp'] < prev_last_candle['vwap_target_smooth_supp']:
    #         #     self.dp.send_msg(f"Aborting entry for pair {pair}. Still downtrending.")
    #         #     return False
    #         if rate > (last_candle["close"] * (1 + 0.0035)):
    #             self.dp.send_msg(f"{pair} - Not entering trade, slippage too high. Last candle close: {last_candle['close']}, Entry: {rate}, Slippage (%): {(rate / last_candle['close']) - 1}")
    #             return False
    #     else:
    #         # if last_candle['vwap_target_smooth_supp'] > prev_last_candle['vwap_target_smooth_supp']:
    #         #     self.dp.send_msg(f"Aborting entry for pair {pair}. Still uptrending.")
    #         #     return False
    #         if rate < (last_candle["close"] * (1 - 0.0035)):
    #             self.dp.send_msg(f"{pair} - Not entering trade, slippage too high. Last candle close: {last_candle['close']}, Entry: {rate}, Slippage (%): {(rate / last_candle['close']) - 1}")
    #             return False
    #
    #     return True

    #def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']
        timeframe = self.timeframe

        for producer in self._producers:

            producer_timeframe = self._producer_tfs[producer]
            producer_dataframe, _ = self.dp.get_producer_df(pair, timeframe=producer_timeframe, producer_name=producer)

            if not producer_dataframe.empty:

                dataframe = merge_informative_pair(dataframe, producer_dataframe, timeframe,
                                                    producer_timeframe,
                                                    append_timeframe=False,
                                                    suffix=producer, ffill=False)

            else:
                dataframe[self._columns_to_expect[producer]] = np.nan

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the entry signal for the given dataframe
        """
        # Use the dataframe columns as if we calculated them ourselves

        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        conditions = []

        Consumer1 = (
            (dataframe['enter_long_Consumer'] == 1) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[Consumer1, 'enter_tag'] += f'{dataframe["enter_tag_Consumer1"]}; '
        conditions.append(Consumer1)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.sell_params['pHSL']
        PF_1 = self.sell_params['pPF_1']
        SL_1 = self.sell_params['pSL_1']
        PF_2 = self.sell_params['pPF_2']
        SL_2 = self.sell_params['pSL_2']

        # dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # current_candle = dataframe.iloc[-1].squeeze()
        # current_profit = trade.calc_profit_ratio(current_candle['close'])

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if self.can_short:
            if (-1 + ((1 - sl_profit) / (1 - current_profit))) <= 0:
                return 1
        else:
            if (1 - ((1 + sl_profit) / (1 + current_profit))) <= 0:
                return 1

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

class Consumer5_dca(Consumer5):

    # DCA options
    position_adjustment_enable = True
    initial_safety_order_trigger = -0.01
    max_safety_orders = 14
    safety_order_step_scale = 1         #SS
    safety_order_volume_scale = 1.05        #OS
    multiplier = 2 #BO:SO ratio
    ### COMMENT SMIDELIS: Add additional variable for BO/SO ratio as "multiplier" here
    ### COMMENT SMIDELIS: The below lines 239 - 245 are calculating a multiplier which is used below in "custom_stake_amount" to calculate your stake amount. As you can see it's proposed_stake / self.max_dca_multiplier. I guess that only the below formula needs to be adapted to include a BO/SO ratio multiplier. I think that's "already" it.

    # Auto compound calculation
    max_dca_multiplier = (1 + max_safety_orders)
    if (max_safety_orders > 0):
        if (safety_order_volume_scale > 1):
            max_dca_multiplier = (2 + (safety_order_volume_scale * (math.pow(safety_order_volume_scale, (max_safety_orders - 1)) - 1) / (safety_order_volume_scale - 1)))
        elif (safety_order_volume_scale < 1):
            max_dca_multiplier = (2 + (safety_order_volume_scale * (1 - math.pow(safety_order_volume_scale, (max_safety_orders - 1))) / (1 - safety_order_volume_scale)))

    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:

        if self.config['stake_amount'] == 'unlimited':
            return proposed_stake / (self.max_dca_multiplier / 2)

        return proposed_stake

    # DCA
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        filled_entries = trade.select_filled_orders(trade.entry_side) #from stash86
        count_of_entries = len(filled_entries) #from stash86
        count_of_buys = trade.nr_of_successful_buys
        for i in range(len(filled_entries)):
            i = i-1
            logger.info(f'Entry - Pair: {i} - {trade.pair}; Count filled entries: {len(filled_entries)}')
            logger.info(f'Entry - Pair: {i} - {trade.pair}; First filled entry: {filled_entries[i]}')
            logger.info(f'Entry - Pair: {i} - {trade.pair}; Cost: {filled_entries[i].cost}; Amount: {filled_entries[i].amount}; Price: {filled_entries[i].price}; Average: {filled_entries[i].average}')
            logger.info(f'Entry - Pair: {i} - {trade.pair}; First filled entry cost: {filled_entries[i].cost}')
            logger.info(f'Entry - Pair: {i} - {trade.pair}; Open rate: {trade.open_rate}; Current Rate: {current_rate}')
            logger.info(f'Entry - Pair: {i} - {trade.pair}; Current profit: {current_profit}')
            logger.info(f'BO/Current Price: {(((filled_entries[0].price / current_rate) -1) * -1)}')

        if (((filled_entries[0].price / current_rate) -1) * -1) > self.initial_safety_order_trigger:
            return None

        if 1 <= count_of_buys <= self.max_safety_orders:
            safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
            if (self.safety_order_step_scale > 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
            elif (self.safety_order_step_scale < 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (1 - math.pow(self.safety_order_step_scale,(count_of_buys - 1))) / (1 - self.safety_order_step_scale))

            if (((filled_entries[0].price / current_rate) -1) * -1) <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    ### COMMENT SMIDELIS: The below lines 279 and 281 seem to calculate BO and SOs. Might need to be adapted too. It's also referring to "max_dca_multiplier" mentioned above. So the formula for max_dca_multiplier needs to include the BO/SO ratio somehow. Line 281 is the SOs, they might stay like they are, but im not sure.
                    # This calculates base order size
                    stake_amount = stake_amount / (self.max_dca_multiplier / 2)
                    # This then calculates current safety order size
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                    if (count_of_buys > 0): ### Checks if count_of_buys is above 0
                        stake_amount = stake_amount / 2
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None

        return None
