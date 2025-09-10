
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone



class BbStage(IStrategy):
    minimal_roi = {
        "0": 20
    }

    stoploss = -0.2

    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.15
    trailing_only_offset_is_reached = True

    ticker_interval = '1h'

    use_sell_signal = True
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['bb_middleband']) &
                    (dataframe['close'] < dataframe['bb_upperband']) &
                    (dataframe['close'] > dataframe['ema9']) &
                    (dataframe['close'] > dataframe['ema200']) &
                    (dataframe['ema20'] > dataframe['ema200'])

            ),
            'buy'] = 1

        if self.config['runmode'].value in ('live', 'dry_run'):

            trades = Trade.get_trades([Trade.pair == metadata['pair'],
                Trade.open_date > datetime.utcnow() - timedelta(hours=4),
                Trade.is_open.is_(False),
                ]).all()

        sumprofit = sum(trade.close_profit for trade in trades)
        percentprofit = sumprofit / self.config['stake_amount']
        if percentprofit > 0.10:

            self.lock_pair(metadata['pair'], until=datetime.now(timezone.utc) +
                    timedelta(hours=4))

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
                (
                    (dataframe['rsi'] > 75) |
                    (dataframe['close'] < dataframe['bb_middleband'] * 0.97) &
                    (dataframe['open'] > dataframe['close'])  # red bar

                    ),
                'sell'] = 1
        return dataframe

