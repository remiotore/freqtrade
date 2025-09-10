# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
# --------------------------------



class Slowbro(IStrategy):

    minimal_roi = {
         "0": 0.10,
         "1440": 1
    }

    # Stoploss:
    stoploss = -0.99

    timeframe = '1h'
    inf_timeframe = '1d'

    use_sell_signal = False
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30
    process_only_new_candles = False

    def informative_pairs(self):
        # add all whitelisted pairs on informative timeframe
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['30d-low'] = informative['close'].rolling(30).min()

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe[f"30d-low_{self.inf_timeframe}"])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
  
        dataframe['sell'] = 0

        return dataframe
