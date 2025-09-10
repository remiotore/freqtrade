
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class AdxSmasSCSL(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AdxSmas.cs

    """

    INTERFACE_VERSION: int = 3

    can_short: bool = True



    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.25

    timeframe = '1h'

    use_custom_stoploss = True
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()
        
        if current_profit < 0.001 and current_time - timedelta(minutes=140) > trade.open_date_utc:
            return -0.005

        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['long'] = ta.SMA(dataframe, timeperiod=6)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
        dataframe.loc[(), ['enter_long', 'enter_tag']] = (0, 'no_long_enter')

        dataframe.loc[
            (
                    (dataframe['adx'] < 25) &
                    (qtpylib.crossed_above(dataframe['long'], dataframe['short']))

            ),
            'enter_short'] = 1


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'no_long_exit')

        dataframe.loc[
            (
                    (dataframe['adx'] > 25) &
                    (qtpylib.crossed_above(dataframe['short'], dataframe['long']))

            ),
            'exit_short'] = 1

        return dataframe