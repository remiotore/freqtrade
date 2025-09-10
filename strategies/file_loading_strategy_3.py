import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from pandas import DataFrame
from custom_order_form_handler import OrderStatus, StrategyDataHandler
from freqtrade.strategy.interface import IStrategy



class file_loading_strategy_3(IStrategy):

    stoploss = -1.0

    def __init__(self, config) -> None:
        super().__init__(config)
        
        strategy_name = self.__class__.__name__
        self.strategy_name = strategy_name
        self.order_handler = StrategyDataHandler(strategy_name=strategy_name)

    def input_strategy_data(self, pair: str):

        raise NotImplementedError
    
    def set_entry_signal(self, pair: str, dataframe: DataFrame, data: Dict[str, Any]):

        raise NotImplementedError

    
    def get_file_data(self, pair) -> (Dict[str, Any], OrderStatus): # type: ignore

        if pair in self.args:
            d = self.args[pair] 

            return d['data'], d['status']
        else:

            return {}, None
        

    def get_dfile_arg(self, pair, key, ):
        data, status = self.get_file_data(pair)

        
        if key in data:    
            return data[key]
        else:

            return None
    

    
    def set_dfile_arg(self, pair, key, value):

        self.args = self.order_handler.read_strategy_data()
        self.args[pair]["data"][key] = value
        self.order_handler.save_strategy_data(self.args)
   

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Used to read order details from a file and set strategy variables accordingly.
        """


            
        self.args = self.order_handler.read_strategy_data()
                       

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        strategy_data = self.order_handler.read_strategy_data()







        if pair in strategy_data and strategy_data[pair]['status'] == OrderStatus.PENDING.value:
            self.set_entry_signal(pair, dataframe, strategy_data[pair]['data'])
            self.order_handler.update_strategy_data(pair, strategy_data[pair]['data'], OrderStatus.HOLDING)
        else:
            self.set_no_entry(dataframe)
            
        return dataframe

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,rate: float, time_in_force: str, exit_reason: str,current_time: datetime, **kwargs) -> bool:



        strategy_data = self.order_handler.read_strategy_data()
        self.order_handler.update_strategy_data(pair, strategy_data[pair]['data'], OrderStatus.EXITED)
        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def set_no_entry(self, dataframe):
        dataframe.loc[dataframe.index[-1],
                      ['enter_long', 'enter_tag']] = (0, "no_enter")

    def set_no_exit(self, dataframe):
        dataframe.loc[dataframe.index[-1],
                      ['exit_long', 'exit_tag']] = (0, "no_exit")

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,proposed_stake: float, min_stake: Optional[float], max_stake: float,leverage: float, entry_tag: Optional[str], side: str,**kwargs) -> float:

        default_stake = 10 # 10$ if no stake is found
        try:
            return self.get_dfile_arg(pair, 'stake_amount')

        except ValueError as e:
            print(f"Error: {e}")

        return default_stake