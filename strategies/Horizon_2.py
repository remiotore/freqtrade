"""
Horizon_2
加强版正手策略

"""

import logging
from typing import Literal
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, CategoricalParameter,stoploss_from_open
from freqtrade.persistence import Trade,Order
from pandas import DataFrame
import requests
import talib.abstract as ta
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Horizon_2(IStrategy):
    """static config"""
    timeframe = '15m'
    can_short = True
    use_custom_stoploss = True
    stoploss = -0.99  # 99% stoploss
    minimal_roi = {
        "0": 0.01 # 1% ROI
    }
    leverage_opt = 3
    max_entry_position_adjustment = 2
    # This number is explained a bit further down
    max_dca_multiplier = 4.5
    # trailing_stop = True  # This should be a direct boolean
    # trailing_only_offset_is_reached = True # This should be a direct boolean
    """opt value"""
    
    rsi_length = IntParameter(low=7, high=30, default=14, space='buy', optimize=True)
    rsi_buy_threshold = IntParameter(low=10, high=50, default=30, space='buy', optimize=True)
    rsi_sell_threshold = IntParameter(low=50, high=90, default=70, space='sell', optimize=True)
    
    fast_ma_length = IntParameter(low=5, high=20, default=8, space='buy', optimize=True)
    slow_ma_length = IntParameter(low=60, high=120, default=80, space='buy', optimize=True)
    fast_mv_length = IntParameter(low=5, high=20, default=8, space='buy', optimize=True)
    slow_mv_length =  IntParameter(low=40, high=100, default=60, space='buy', optimize=True)
    atr_length = IntParameter(low=40, high=100, default=60, space='buy', optimize=True)
    
    first_profit_rate=DecimalParameter(low=0.01,high=0.04,default=0.02,decimals=2,space='pr',optimize=True)
    second_profit_rate=DecimalParameter(low=0.07,high=0.12,default=0.09,decimals=2,space='pr',optimize=True)
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        
    def leverage(
        self, pair: str, current_time, current_rate: float, proposed_leverage: float, max_leverage: float,
        side: str, **kwargs
    ) -> float:
        return self.leverage_opt  # Use the hyperopt value for leverage

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate technical indicators: SMA (fast/slow), RSI.
        """
        
        dataframe['fast_sma'] = ta.SMA(dataframe, timeperiod=int(self.fast_ma_length.value))
        dataframe['slow_sma'] = ta.SMA(dataframe, timeperiod=int(self.slow_ma_length.value))
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=int(self.rsi_length.value))
        dataframe['fast_mv']=ta.SMA(dataframe['volume'],timeperiod=int(self.fast_mv_length.value))
        dataframe['slow_mv']=ta.SMA(dataframe['volume'],timeperiod=int(self.slow_mv_length.value))
        # dataframe['atr']=ta.ATR(dataframe,14) # type: ignore
        
        

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determine when to open a long or short position based on SMA and RSI signals.
        """
        # 快速ma高于慢速ma，快速mv高于慢速mv，同时rsi处于非超买值，则购买
        long_conditions = (
            (dataframe['fast_sma'] > dataframe['slow_sma']) &
            (dataframe['rsi'].iloc[-1] < self.rsi_buy_threshold.value) &
            (dataframe['fast_mv'] > dataframe ['slow_mv']) 
            
        )
        # 快速ma低于慢速ma，快速mv高于慢速mv，同时rsi处于非超卖值，则卖出
        # 
        short_conditions = (
            (dataframe['fast_sma'] < dataframe['slow_sma']) &
            (dataframe['rsi'].iloc[-1] > self.rsi_sell_threshold.value) &
            (dataframe['fast_mv'] < dataframe ['slow_mv'])
        )

        dataframe.loc[long_conditions, 'enter_long'] = 1
        dataframe.loc[short_conditions, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determine when to exit a long or short position based on reversed conditions.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        long_exit_conditions = (
            (dataframe['fast_sma'] < dataframe['slow_sma']) &
            (dataframe['rsi'].iloc[-1]  > self.rsi_sell_threshold.value) &
            (dataframe['fast_mv'] < dataframe ['slow_mv'])
        )

        short_exit_conditions = (
            (dataframe['fast_sma'] > dataframe['slow_sma']) &
            (dataframe['rsi'].iloc[-1]  < self.rsi_buy_threshold.value) &
            (dataframe['fast_mv'] > dataframe ['slow_mv'])
        )

        dataframe.loc[long_exit_conditions, 'exit_long'] = 1
        dataframe.loc[short_exit_conditions, 'exit_short'] = 1

        return dataframe

    def custom_stoploss( # type: ignore
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> (float | None ):  

        
        # 追踪止损平仓，可以在不同时期使用止损
        count_of_entries = trade.nr_of_successful_entries
        
        if count_of_entries==0:
            #  宽松止盈
            return -0.1
        elif count_of_entries==1:
            # 中等限度的止盈
            return -0.5
        elif count_of_entries==2:
            # 不容忍下跌的止盈
            return -0.2
        
        return float(self.stoploss)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> float | None | tuple[float | None, str | None]:
        if trade.has_open_orders:
            return

        if current_profit > self.first_profit_rate.value and trade.nr_of_successful_exits == 0:
             
            # 此时要进行加仓，但加仓具有阶梯性，我认为，总共是两次加仓
            # 加仓

            filled_entries = trade.select_filled_orders(trade.entry_side)
            count_of_entries = trade.nr_of_successful_entries

            try:
                # This returns first order stake size
                stake_amount = filled_entries[0].stake_amount_filled
                # This then calculates current safety order size
                stake_amount = stake_amount * (1 + (count_of_entries * 0.5))
                return stake_amount, "正手第一次加仓"
            except Exception as exception:
                return None
        
        if current_profit > self.second_profit_rate.value and trade.nr_of_successful_exits != 0:
            # 假设当下利润率达到9，并且已经加过一次仓位
            filled_entries = trade.select_filled_orders(trade.entry_side)
            count_of_entries = trade.nr_of_successful_entries

            try:
                # This returns first order stake size
                stake_amount = filled_entries[0].stake_amount_filled
                # This then calculates current safety order size
                stake_amount = stake_amount * (1 + (count_of_entries * 0.5))
                return stake_amount, "正手第二次加仓"
            except Exception as exception:
                return None

        # if current_profit > -0.05:
        #     # 当且仅当在利润幅度在5%以上时加仓，否则不进行加仓
        #     return -(trade.stake_amount / 2), "half_profit_5%"
        #     return None

        

        return None

    @property
    def protections(self): # type: ignore
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 4,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": 0.1
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 2,
                "stop_duration_candles": 12,
                "only_per_pair": False
            },

        ]
    
    
    
   

    def bot_start(self, **kwargs) -> None:
        """
        Called at bot start. Use this to assign the hyperopt values to the actual parameters.
        """
        super().bot_start(**kwargs)
    
