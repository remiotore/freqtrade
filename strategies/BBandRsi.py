
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, DecimalParameter, merge_informative_pair)

import logging
logger = logging.getLogger(__name__)



class BBandRsi(IStrategy):



    minimal_roi = {
        "0": 10
    }

    stoploss = -0.325


    trailing_stop = True
    trailing_stop_positive = 0.224
    trailing_stop_positive_offset = 0.319
    trailing_only_offset_is_reached = True
    use_custom_stoploss = False

    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    dca_stake_decrease_percent = DecimalParameter(0.5, 1, decimals=2, default=0.89, space="buy")
    dca_stake_adj = DecimalParameter(0.33, 0.95, decimals=2, default=0.63, space="buy")

    dca_num_buys = 2

    @property
    def plot_config(self):
        return {

            'main_plot': {




                'sar_4h': {'color': 'blue'},
                'smma12_4h': {'color': 'lightgreen'},

            },
            'subplots': {

                "RSI": {
                    'rsi_4h': {'color': 'red'},
                }
            }
        }

    stoploss = -0.99

    timeframe = '15m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['avg_price'] = ((dataframe['high'] + dataframe['low'])/2)

        dataframe['sar'] = ta.SAR(dataframe)

        inf_tf = '4h'

        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=inf_tf)



        informative['avg_price_last_day'] = (
            (informative['high'] + informative['low'])/2)

        informative['sar'] = ta.SAR(informative)



        informative['smma12'] = ta.SMA(ta.SMA(informative, timeperiod=12), 12)

        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['smma12_4h'] < dataframe['avg_price']) &
                (dataframe['rsi'] < 30) &
                (dataframe['close'] < dataframe['bb_lowerband'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        conditions = []








        conditions.append((dataframe['smma12_4h'] < dataframe['avg_price']))


        conditions.append((dataframe['volume'] > 0))
        conditions.append((qtpylib.crossed_below(
            dataframe['sar_4h'], dataframe['avg_price'])))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def custom_stake_amount(self, pair: str, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:


        new_proposed_stake = (proposed_stake*self.dca_stake_adj.value)
        logger.debug("Proposed Stake: %s", new_proposed_stake)

        return new_proposed_stake


    def adjust_trade_position(self, trade: Trade, current_time: 'datetime',
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(
            trade.pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        count_of_buys = trade.nr_of_successful_buys
        open_order_price = trade.open_rate

        if ((last_candle['close'] > (open_order_price * self.dca_stake_decrease_percent.value))):
            return None
        if (count_of_buys > self.dca_num_buys):
            return None

        filled_buys = trade.select_filled_orders('buy')
        num_filled_buys = len(filled_buys)





        try:

            stake_amount = filled_buys[0].cost
            new_stake_amount = ((stake_amount/self.dca_stake_adj.value)
                                * ((1-self.dca_stake_adj.value)/self.dca_num_buys))


            logger.debug('\nLast candle date: %s \n\
                            - count of buys: %s \n\
                            - max stake: %s \n\
                            - Stake Amount: %s \n\
                            - New Stake Amount: %s \n',
                         last_candle['date'],
                         count_of_buys,
                         max_stake,
                         stake_amount,
                         new_stake_amount)
            return (new_stake_amount)
        except Exception as exception:
            return None

        return None
