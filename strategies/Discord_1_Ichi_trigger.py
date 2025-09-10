# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --------------------------------
import pandas as pd
import numpy as np
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes


class Ichi_trigger(IStrategy):

    # Do not change the timeframe values in this file
    # Use the config file or command line options to select the appropriate timeframe:
    #     5m - BACKTEST or HYPEROPT
    #     1h - LIVE or DRYRUN

    # Backtest or hyperopt at this timeframe
    timeframe = '5m'
    # Generate signals from the 1h timeframe
    # Live or Dry-run at this timeframe
    informative_timeframe = '1d'

    # WARNING
    # ichimoku is a long indicator, if you remove or use
    # shorter startup_candle_count your results will be unstable/invalid
    # for up to a week from the start of your backtest or dry/live run
    # (180 candles = 7.5 days)
    startup_candle_count = 120

    # This strat only uses candle information, so processing between
    # new candles is a waste of resources as nothing will change
    process_only_new_candles = True

    # ROI table:
    minimal_roi = {
         "0": 9.99
        # "0": 0.16,
        # "40": 0.1,
        # "90": 0.035,
        # "210": 0
    }

    # I haven't been able to determine a good default stoploss.
    # Select or hyperopt an stoploss that you're happy with, and backtest the result.
    #
    # Do not use stoploss_on_exchange if you leave the stoploss at the default value
    # or the bot may trigger emergencysell when it fails to place the stoploss.
    #
    # Stoploss:
    stoploss = -0.99

    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(0,0,0,0.2)',
            },
            # plot senkou_b, too. Not only the area to it.
            'senkou_b': {
                'color': 'red',
            },
            'tenkan_sen': { 'color': 'orange' },
            'kijun_sen': { 'color': 'blue' },

            'chikou_span': { 'color': 'lightgreen' },

            # 'ssl_up': { 'color': 'green' },
            # 'ssl_down': { 'color': 'red' },
        },
        'subplots': {
            "Signals": {
                'go_long': {'color': 'blue'},
                'buy_criteria': {'color': 'green'},
                'sell_criteria': {'color': 'red'},
            },
        }
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs
    def do_indicators(self, dataframe: DataFrame, metadata: dict):

        # # Standard Settings
        # displacement = 26
        # ichimoku = ftt.ichimoku(dataframe,
        #     conversion_line_period=9,
        #     base_line_periods=26,
        #     laggin_span=52,
        #     displacement=displacement
        #     )

        # Crypto Settings
        displacement = 30
        ichimoku = ftt.ichimoku(dataframe,
            conversion_line_period=20,
            base_line_periods=60,
            laggin_span=120,
            displacement=displacement
            )

        dataframe['chikou_span'] = ichimoku['chikou_span']

        # cross indicators
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

        # cloud, green a > b, red a < b
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        # DANGER ZONE START

        # NOTE: Not actually the future, present data that is normally shifted forward for display as the cloud
        dataframe['future_green'] = (dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b']).astype('int') * 2

        # The chikou_span is shifted into the past, so we need to be careful not to read the
        # current value.  But if we shift it forward again by displacement it should be safe to use.
        # We're effectively "looking back" at where it normally appears on the chart.
        dataframe['chikou_high'] = (
                (dataframe['chikou_span'] > dataframe['senkou_a']) &
                (dataframe['chikou_span'] > dataframe['senkou_b'])
            ).shift(displacement).fillna(0).astype('int')

        #END DANGER ZONE

        dataframe['kumo_high'] = max(dataframe['leading_senkou_span_a'].shift(displacement-1), dataframe['leading_senkou_span_b'].shift(displacement-1))
        dataframe['kumo_low'] = min(dataframe['leading_senkou_span_a'].shift(displacement-1), dataframe['leading_senkou_span_b'].shift(displacement-1))

        if dataframe['tenkan_sen'] > dataframe['kumo_high']:
            tkkh_sig_w = 1
        elif dataframe['tenkan_sen'] < dataframe['kumo_low']:
            tkkh_sig_w = 0


        if dataframe['close'] > dataframe['high']:
            csh_sig_w = 1
        elif dataframe['close'] < dataframe['low'].shift(25):
            csh_sig_w = 0

        if dataframe['close'] > dataframe['kumo_high'].shift(displacement-1):
            cskh_sig_w = 1
        elif dataframe['close'] < dataframe['kumo_low'].shift(displacement-1):
            cskh_sig_w = 0
        if dataframe['close'] > dataframe['tenkan_sen']:
            ptk_sig_w = 1
        elif dataframe['close'] < dataframe['tenkan_sen']:
            ptk_sig_w = 0

        if dataframe['close'] > dataframe['kijun_sen']:
            pkj_sig_w = 1
        elif dataframe['close'] < dataframe['kijun_sen']:
            pkj_sig_w = 0

        if dataframe['tenkan_sen'] > dataframe['kijun_sen']:
           tkkj_sig_w = 1
        elif dataframe['tenkan_sen'] < dataframe['kijun_sen']:
           tkkj_sig_w = 0

        if dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b']:
           sasb_sig_w = 1
        elif dataframe['leading_senkou_span_a'] < dataframe['leading_senkou_span_b']:
           sasb_sig_w = 0
        if dataframe['close'] > dataframe['kumo_high']:
           ckh_sig_w = 1
        elif dataframe['close'] < dataframe['kumo_low']:
           ckh_sig_w = 0

        dataframe['bs_pic_sig'] = tkkh_sig_w + csh_sig_w + cskh_sig_w + ptk_sig_w + pkj_sig_w + tkkj_sig_w + sasb_sig_w + ckh_sig_w

        dataframe['pic_l_s_trig'] =  5.1


        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

       # if self.config['runmode'].value in ('backtest', 'hyperopt'):
       #     assert (timeframe_to_minutes(self.timeframe) <= 5), "Backtest this strategy in 5m or 1m timeframe. Read comments for details."

        if self.timeframe == self.informative_timeframe:
            dataframe = self.do_indicators(dataframe, metadata)
        else:
            if not self.dp:
                return dataframe

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

            informative = self.do_indicators(informative.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
            # don't overwrite the base dataframe's HLCV information
            skip_columns = [(s + "_" + self.informative_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s, inplace=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
                (dataframe['bs_pic_sig'] > dataframe['pic_l_s_trig'])
        ,
        'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
                    (dataframe['bs_pic_sig'] < dataframe['pic_l_s_trig'])
        ,
        'sell'] = 1

        return dataframe
