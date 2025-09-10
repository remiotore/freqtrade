import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, merge_informative_pair

## Missing IMports
from pandas import DataFrame

## DCA Imports
import math
import logging

logger = logging.getLogger(__name__)

class Consumer(IStrategy):
    timeframe = '1m'
    sell_params = {
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.32,
        "pPF_1": 0.02,
        "pPF_2": 0.047,
        "pSL_1": 0.02,
        "pSL_2": 0.046,
    }

    #...
    process_only_new_candles = False # required for consumers
    use_custom_stoploss = True
    stoploss = -0.999
    #### Comment: Dont know if this is right, you need to test how it needs to be configured for multiple producers
    _columns_to_expect = ['enter_long_NFIX', 'enter_long_Cluc', 'enter_long_BB_RPB', 'enter_long_Elliot']

    #### Comment: Your timeperiod is 1 all the time. I guess this should be adapted to the timeperiod used by 3c
    @informative('3m')
    def populate_indicators_3m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['uo'] = ta.ULTOSC(dataframe, timeperiod=1)

        return dataframe

    @informative('2h')
    def populate_indicators_2h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['uo'] = ta.ULTOSC(dataframe, timeperiod=1)

        return dataframe

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Use the websocket api to get pre-populated indicators from another freqtrade instance.
        Use `self.dp.get_producer_df(pair)` to get the dataframe
        """
        pair = metadata['pair']
        timeframe = self.timeframe

        producer_pairs = self.dp.get_producer_pairs()
        # You can specify which producer to get pairs from via:
        # self.dp.get_producer_pairs("my_other_producer")

        #### Comment: Producer 1
        # This func returns the analyzed dataframe, and when it was analyzed
        producer_dataframe_NFIX, _ = self.dp.get_producer_df(pair, producer_name="NFIX")
        # You can get other data if the producer makes it available:
        # self.dp.get_producer_df(
        #   pair,
        #   timeframe="1h",
        #   candle_type=CandleType.SPOT,
        #   producer_name="my_other_producer"
        # )

        if not producer_dataframe_NFIX.empty:
            # If you plan on passing the producer's entry/exit signal directly,
            # specify ffill=False or it will have unintended results
            merged_dataframe = merge_informative_pair(dataframe, producer_dataframe_NFIX,
                                                      timeframe, timeframe,
                                                      append_timeframe=False,
                                                      suffix="NFIX", ffill=False)
            return merged_dataframe
        else:
            dataframe[self._columns_to_expect] = 0

        #### Comment: Producer 2
        # This func returns the analyzed dataframe, and when it was analyzed
        producer_dataframe_Cluc, _ = self.dp.get_producer_df(pair, producer_name="Cluc")
        # You can get other data if the producer makes it available:
        # self.dp.get_producer_df(
        #   pair,
        #   timeframe="1h",
        #   candle_type=CandleType.SPOT,
        #   producer_name="my_other_producer"
        # )

        if not producer_dataframe_Cluc.empty:
            # If you plan on passing the producer's entry/exit signal directly,
            # specify ffill=False or it will have unintended results
            merged_dataframe = merge_informative_pair(dataframe, producer_dataframe_Cluc,
                                                      timeframe, timeframe,
                                                      append_timeframe=False,
                                                      suffix="Cluc", ffill=False)
            return merged_dataframe
        else:
            dataframe[self._columns_to_expect] = 0

        #### Comment: Producer 3
        # This func returns the analyzed dataframe, and when it was analyzed
        producer_dataframe_BB_RPB, _ = self.dp.get_producer_df(pair, producer_name="BB_RPB")
        # You can get other data if the producer makes it available:
        # self.dp.get_producer_df(
        #   pair,
        #   timeframe="1h",
        #   candle_type=CandleType.SPOT,
        #   producer_name="my_other_producer"
        # )

        if not producer_dataframe_BB_RPB.empty:
            # If you plan on passing the producer's entry/exit signal directly,
            # specify ffill=False or it will have unintended results
            merged_dataframe = merge_informative_pair(dataframe, producer_dataframe_BB_RPB,
                                                      timeframe, timeframe,
                                                      append_timeframe=False,
                                                      suffix="BB_RPB", ffill=False)
            return merged_dataframe
        else:
            dataframe[self._columns_to_expect] = 0

        #### Comment: Producer 4
        # This func returns the analyzed dataframe, and when it was analyzed
        producer_dataframe_Elliot, _ = self.dp.get_producer_df(pair, producer_name="Elliot")
        # You can get other data if the producer makes it available:
        # self.dp.get_producer_df(
        #   pair,
        #   timeframe="1h",
        #   candle_type=CandleType.SPOT,
        #   producer_name="my_other_producer"
        # )

        if not producer_dataframe_Elliot.empty:
            # If you plan on passing the producer's entry/exit signal directly,
            # specify ffill=False or it will have unintended results
            merged_dataframe = merge_informative_pair(dataframe, producer_dataframe_Elliot,
                                                      timeframe, timeframe,
                                                      append_timeframe=False,
                                                      suffix="Elliot", ffill=False)
            return merged_dataframe
        else:
            dataframe[self._columns_to_expect] = 0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the entry signal for the given dataframe
        """
        # Use the dataframe columns as if we calculated them ourselves

        # ----------------------------
        dataframe["enter_long"] = 0 #Added due to error regarding tables not being initialized
        dataframe["enter_tag"] = 0 #Added due to error regarding tables not being initialized
        dataframe.loc[
            (
                (dataframe['enter_long_Cluc'] == 1) &
                (
                    (dataframe['rsi_3m'] > 70) &
                    (dataframe['uo_3m'] > 67) &
                    (dataframe['uo_2h'] > 57) &
                    (dataframe['rsi_4h'] > 63)
                ) &
                (dataframe['volume'] > 0)
            ), ['enter_long', 'enter_tag']] += (1, 'enter_long_Cluc;')
        # ----------------------------
        dataframe.loc[
            (
                (dataframe['enter_long_NFIX'] == 1) &
                (
                    (dataframe['rsi_3m'] > 70) &
                    (dataframe['uo_3m'] > 67) &
                    (dataframe['uo_2h'] > 57) &
                    (dataframe['rsi_4h'] > 63)
                ) &
                (dataframe['volume'] > 0)
            ), ['enter_long', 'enter_tag']] += (1, 'enter_long_NFIX;')
        # ----------------------------
        dataframe.loc[
            (
                (dataframe['enter_long_BB_RPB'] == 1) &
                (
                    (dataframe['rsi_3m'] > 70) &
                    (dataframe['uo_3m'] > 67) &
                    (dataframe['uo_2h'] > 57) &
                    (dataframe['rsi_4h'] > 63)
                ) &
                (dataframe['volume'] > 0)
            ), ['enter_long', 'enter_tag']] += (1, 'enter_long_BB_RPB;')
        # ----------------------------
        dataframe.loc[
            (
                (dataframe['enter_long_Elliot'] == 1) &
                (
                    (dataframe['rsi_3m'] > 70) &
                    (dataframe['uo_3m'] > 67) &
                    (dataframe['uo_2h'] > 57) &
                    (dataframe['rsi_4h'] > 63)
                ) &
                (dataframe['volume'] > 0)
            ), ['enter_long', 'enter_tag']] += (1, 'enter_long_Elliot;')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        current_profit = trade.calc_profit_ratio(current_candle['close'])

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

class Consumer_dca(Consumer):

    # DCA options
    position_adjustment_enable = True

    initial_safety_order_trigger = -0.01
    max_safety_orders = 30
    safety_order_step_scale = 1         #SS
    safety_order_volume_scale = 1.05        #OS

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
            return proposed_stake / self.max_dca_multiplier

        return proposed_stake

    # DCA
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        if current_profit > self.initial_safety_order_trigger:
            return None

        count_of_buys = trade.nr_of_successful_buys

        if 1 <= count_of_buys <= self.max_safety_orders:
            safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
            if (self.safety_order_step_scale > 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
            elif (self.safety_order_step_scale < 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (1 - math.pow(self.safety_order_step_scale,(count_of_buys - 1))) / (1 - self.safety_order_step_scale))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    # This calculates base order size
                    stake_amount = stake_amount / self.max_dca_multiplier
                    # This then calculates current safety order size
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None

        return None
