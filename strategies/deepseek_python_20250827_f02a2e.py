from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
from datetime import datetime

class Top100TrendStrategy(IStrategy):

    INTERFACE_VERSION: int = 3
    
    # Optimal ROI
    minimal_roi = {
        "0": 0.04,   # 4% with 3x leverage = 12% actual
        "20": 0.03,  # 3% with 3x leverage = 9% actual
        "40": 0.02   # 2% with 3x leverage = 6% actual
    }
    
    stoploss = -0.045
    trailing_stop = False
    leverage_num = 3
    leverage_denom = 1
    can_short = False
    timeframe = "5m"
    max_open_trades = 5

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str, side: str, **kwargs) -> float:
        return 3.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma20'] = ta.SMA(dataframe['close'], timeperiod=20)
        dataframe['sma50'] = ta.SMA(dataframe['close'], timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        dataframe['momentum'] = dataframe['close'].pct_change(periods=3)
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get current pair from metadata
        current_pair = metadata['pair']
        
        # SKIP problematic pairs entirely
        skip_pairs = ['AAVE/USDT:USDT', 'ICP/USDT:USDT', 'ETC/USDT:USDT']
        if current_pair in skip_pairs:
            return dataframe  # No entries for these pairs
        
        very_strong_volume = dataframe['volume_ratio'] > 1.8
        perfect_rsi = (dataframe['rsi'] > 45) & (dataframe['rsi'] < 55)
        strong_trend = dataframe['sma20'] > dataframe['sma50']
        positive_momentum = dataframe['momentum'] > 0
        price_above_smas = (dataframe['close'] > dataframe['sma20']) & (dataframe['close'] > dataframe['sma50'])
        
        dataframe.loc[
            price_above_smas &
            very_strong_volume &
            perfect_rsi &
            strong_trend &
            positive_momentum &
            (dataframe['close'].shift(1) > dataframe['sma20'].shift(1)) &
            (dataframe['volume_ratio'].shift(1) > 1.5),
            'enter_long'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 75),
            'exit_long'] = 1
            
        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        return self.stoploss

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs):
        """Early profit taking for best pairs"""
        best_pairs = ['ATOM/USDT:USDT', 'THETA/USDT:USDT', 'NEAR/USDT:USDT', 'XLM/USDT:USDT']
        
        if pair in best_pairs and current_profit > 0.03:
            return 'early_profit_best_pair'
            
        return None