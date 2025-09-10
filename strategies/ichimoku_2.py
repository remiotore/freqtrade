
import logging
import talib.abstract as ta
import pandas as pd
from pandas import DataFrame
import arrow
from pathlib import Path
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.arguments import TimeRange
from technical.indicator_helpers import fishers_inverse
from freqtrade.strategy.interface import IStrategy
from freqtrade.state import RunMode
from freqtrade.strategy.util import resample_to_interval, resampled_merge
from freqtrade.data.history import parse_ticker_dataframe, load_pair_history
import freqtrade.indicators as indicators

logger = logging.getLogger("IchimokuStrategy")

class ichimoku_2(IStrategy):

    cache = {}
    min_days = 30

    def get_extend_historical(self, pair: str, dataframe: DataFrame) -> DataFrame:

        if hasattr(self, 'dp'):
            if self.dp.runmode in (RunMode.LIVE, RunMode.DRY_RUN):
                min_date = dataframe['date'].min()
                if pair not in self.cache or self.cache[pair]["date"].max() < min_date:
                    logger.info(f"Downloading historical ohlc for pair: {pair})")



                    self.cache[pair] = load_pair_history(pair, 
                                        ticker_interval=self.ticker_interval, 
                                        datadir= Path(f"user_data/data/history"),
                                        timerange=TimeRange(starttype ='date', startts=int(arrow.utcnow().shift(days=-60).float_timestamp)),
                                        refresh_pairs=True,
                                        exchange=self.dp._exchange)

                hist_df = self.cache[pair]
                min_date = dataframe['date'].min()
                return pd.concat([ hist_df[hist_df['date'] < min_date] , dataframe ])

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        
        dataframe = self.get_extend_historical(metadata['pair'], dataframe)
        if (dataframe['date'].max() - dataframe['date'].min()).days < self.min_days:
            return dataframe


        macd = ta.MACD(dataframe, fastperiod=14, slowperiod=27, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)


        dataframe_1d =  resample_to_interval(dataframe, '1d')
        macd = ta.MACD(dataframe_1d, fastperiod=14, slowperiod=27, signalperiod=15)
        dataframe_1d['macd_1d'] = macd['macd']
        dataframe_1d['macdsignal_1d'] = macd['macdsignal']
        dataframe_1d['macdhist_1d'] = macd['macdhist']
        dataframe_1d['sar_1d'] = ta.SAR(dataframe_1d, acceleration=0.15, maximum=1)
        dataframe_1d['close_1d'] = dataframe_1d['close']
        dataframe_1d['rsi_1d'] = ta.RSI(dataframe_1d)


        dataframe_4h =  resample_to_interval(dataframe, '4h')
        ichimoku = indicators.ichimoku(dataframe_4h, tenkan_sen_window=15, kijun_sen_window=27, senkou_span_offset=15, senkou_span_b_window=50)
        dataframe_4h['tenkan_sen_4h'] = ichimoku['tenkan_sen']
        dataframe_4h['kijun_sen_4h'] = ichimoku['kijun_sen']
        dataframe_4h['senkou_span_a_4h'] = ichimoku['senkou_span_a']
        dataframe_4h['senkou_span_b_4h'] = ichimoku['senkou_span_b']
        macd = ta.MACD(dataframe_4h, fastperiod=14, slowperiod=27, signalperiod=9)
        dataframe_4h['macd_4h'] = macd['macd']
        dataframe_4h['macdsignal_4h'] = macd['macdsignal']

        dataframe = resampled_merge(dataframe, dataframe_4h)
        dataframe = resampled_merge(dataframe, dataframe_1d)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if (dataframe['date'].max() - dataframe['date'].min()).days < self.min_days:
            dataframe['buy'] = 0
            return dataframe

        try:
            if self.dp:
                if self.dp.runmode in (RunMode.LIVE, RunMode.DRY_RUN):
                    ticker_data = self.dp._exchange.get_ticker(metadata['pair'])
                    symbol,bid,ask,last= ticker_data['symbol'],ticker_data['bid'],ticker_data['ask'],ticker_data['last']
                    
                    if(ask <=0 or bid <=0 or last <= 0):
                        dataframe['buy'] = 0
                        return dataframe

                    spread = ((ask - bid)/last) * 100
                    if(spread > 0.20):
                        dataframe['buy'] = 0
                        return dataframe
        except:
            print(f"could not get ticker for pair: {metadata['pair']}")
            dataframe['buy'] = 0
            return dataframe

        dataframe.loc[
            (
                (
                    (dataframe['macd'] > dataframe['macdsignal']) &
                    (dataframe['macd'].shift() > dataframe['macdsignal'].shift()) &
                    (dataframe['macd_4h'] > dataframe['macdsignal_4h']) &
                    (dataframe['sar_1d'] < dataframe['open']) &
                    (dataframe['sar'] < dataframe['open']) &

                    (dataframe['macd_1d'] > dataframe['macdsignal_1d']) &
                    (dataframe['rsi'] < 75) &
                    (dataframe['open'] > dataframe['senkou_span_a_4h']) &
                    (dataframe['open'] > dataframe['senkou_span_b_4h']) &
                    (dataframe['close'] > dataframe['senkou_span_a_4h']) &
                    (dataframe['close'] > dataframe['senkou_span_b_4h'])
                )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        if (dataframe['date'].max() - dataframe['date'].min()).days < self.min_days:
            dataframe['sell'] = 0
            return dataframe

        dataframe.loc[
            (

                 (dataframe['sar_1d'] > dataframe['open']) 
            ),
            'sell'] = 1
        return dataframe
