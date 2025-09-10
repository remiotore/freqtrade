from datetime import datetime, timedelta
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import IntParameter, DecimalParameter, CategoricalParameter
import numpy as np
import logging

class hungry2(IStrategy):

    buy_threshold = DecimalParameter(0.01, 0.05, default=0.02)
    buy_velocity = DecimalParameter(0.0001, 0.01, default=0.001)
    additional_buy = DecimalParameter(0.01, 0.1, default=0.05)
    sell_threshold = DecimalParameter(0.01, 0.1, default=0.05)
    loss_velocity = DecimalParameter(0.0001, 0.01, default=0.001)
    loss_percentage = DecimalParameter(0.01, 0.1, default=0.03)
    sell_quantity = DecimalParameter(0.1, 1.0, default=0.5)
    total_loss = DecimalParameter(0.05, 0.2, default=0.1)
    recovery_growth = DecimalParameter(0.01, 0.1, default=0.05)
    recovery_duration = IntParameter(10, 60, default=30)
    stoploss = DecimalParameter(-0.3, -0.01, default=-0.1, space='stoploss')

    use_trailing_stop = CategoricalParameter([True, False], default=True)
    trailing_stop_distance = DecimalParameter(0.01, 0.05, default=0.02)

    logger = logging.getLogger(__name__)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.stoploss = self.stoploss.value
        self.trailing_stop = self.use_trailing_stop.value
        self.trailing_stop_positive = self.trailing_stop_distance.value
        self.locked_states = []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['recovery'] = dataframe['close'].pct_change(self.recovery_duration.value)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        locked_mask = np.array(self.locked_states + [0] * (len(dataframe) - len(self.locked_states)))
        
        conditions = (
            (locked_mask == 0) &
            (dataframe['close'] > dataframe['close'].shift(1) * (1 + self.buy_threshold.value)) &
            (dataframe['close'] - dataframe['close'].shift(1) > self.buy_velocity.value)
        )
        
        dataframe['enter_long'] = np.where(conditions, 1 + self.additional_buy.value, 0)

        locked_states_conditions = (dataframe['close'] < dataframe['close'].shift(1) * (1 - self.total_loss.value))
        self.locked_states = np.where(locked_states_conditions, 1, 0).tolist()

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        locked_mask = np.array(self.locked_states + [0] * (len(dataframe) - len(self.locked_states)))
        
        conditions_up = dataframe['close'] > dataframe['open'] * (1 + self.sell_threshold.value)
        conditions_down = (
            ((dataframe['close'] - dataframe['close'].shift(1)) < self.loss_velocity.value) &
            (dataframe['close'] < dataframe['open'] * (1 - self.loss_percentage.value))
        )
        conditions_total_loss = dataframe['close'] < dataframe['open'] * (1 - self.total_loss.value)

        dataframe['exit_long'] = np.where(conditions_up | conditions_down | conditions_total_loss, 1, 0)

        locked_states_conditions = (dataframe['close'] < dataframe['close'].shift(1) * (1 - self.total_loss.value))
        self.locked_states = np.where(locked_states_conditions, 1, 0).tolist()

        return dataframe
