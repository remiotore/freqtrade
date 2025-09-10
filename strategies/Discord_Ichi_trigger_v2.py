# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --------------------------------
import pandas as pd
import numpy as np
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes

#
# NAVI_Ichi_trigger v1 - 07/10/2021
#
# Based on Obelisk_TradePro_Ichi v2.2 by Obelisk
# https://github.com/brookmiles/
#

class Ichi_trigger_v2(IStrategy):


    # Backtest or hyperopt at this timeframe
    timeframe = '5m'

    # Generate signals from the 1d timeframe
    # Live or Dry-run at this timeframe
    informative_timeframe = '1h'

    startup_candle_count = 60

    process_only_new_candles = True

    # ROI table:
    minimal_roi = {
         "0": 9.99
        # "0": 0.16,
        # "40": 0.1,
        # "90": 0.035,
        # "210": 0
    }

    # Stoploss:
    stoploss = -0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def do_indicators(self, dataframe: DataFrame, metadata: dict):

        displa = 26-1
        displacement = 26
        ichimoku = ftt.ichimoku(dataframe,
            conversion_line_period=9,
            base_line_periods=26,
            laggin_span=52,
            displacement=displacement
            )


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

        #DEFINE KUMO 'NOW'
        lsa = dataframe['leading_senkou_span_a']
        lsb = dataframe['leading_senkou_span_b']
        dataframe['kumo_high'] = pd.DataFrame(np.where(lsa > lsb, lsa, lsb))
        dataframe['kumo_low'] = pd.DataFrame(np.where(lsa < lsb, lsa, lsb))

        # TENKAN - KUMO
        dataframe['tkkh_sig_w'] = 0
        dataframe['tkkh_sig_w'] = np.select(condlist=[dataframe['tenkan_sen'] > dataframe['kumo_high'],
                                  dataframe['tenkan_sen'] < dataframe['kumo_low']],
                                  choicelist=[1,0], default=0)

        # PRICE - CHIKOU HIGH-LOW
        dataframe['csh_sig_w'] = 0
        dataframe['csh_sig_w'] = np.select(condlist=[dataframe['close'] > dataframe['high'].shift(displa),
                                 dataframe['close'] < dataframe['low'].shift(displa)],
                                 choicelist=[1,0], default=0)
        # PRICE - TENKAN-SEN
        dataframe['ptk_sig_w'] = 0
        dataframe['ptk_sig_w'] = np.select(condlist=[dataframe['close'] > dataframe['tenkan_sen'],
                                 dataframe['close'] < dataframe['tenkan_sen']],
                                 choicelist=[1,0], default=0)

        # PRICE - KIJUN-SEN
        dataframe['pkj_sig_w'] = 0
        dataframe['pkj_sig_w'] = np.select(condlist=[dataframe['close'] > dataframe['kijun_sen'],
                                 dataframe['close'] < dataframe['kijun_sen']],
                                 choicelist=[1,0], default=0)

        # TENKAN-SEN / KIJUN-SEN
        dataframe['tkkj_sig_w'] = 0
        dataframe['tkkj_sig_w'] = np.select(condlist=[dataframe['tenkan_sen'] > dataframe['kijun_sen'],
                                 dataframe['tenkan_sen'] < dataframe['kijun_sen']],
                                 choicelist=[1,0], default=0)

        # NUBE EN EL FUTURO (PROYECCION)
        dataframe['sasb_sig_w'] = 0
        dataframe['sasb_sig_w'] = np.select(condlist=[dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b'],
                                 dataframe['leading_senkou_span_a'] < dataframe['leading_senkou_span_b']],
                                 choicelist=[1,0], default=0)

        # PRICE CHIKOU - KUMO
        dataframe['ckh_sig_w'] = 0
        dataframe['ckh_sig_w'] = np.select(condlist=[dataframe['close'] > dataframe['kumo_high'],
                                 dataframe['close'] < dataframe['kumo_low']],
                                 choicelist=[1,0], default=0)
        dataframe['bs_pic_sig'] = reduce(lambda a, b: a.add(b, fill_value=0),
                                  [
                                  dataframe['tkkh_sig_w'], dataframe['csh_sig_w'], dataframe['cskh_sig_w'], dataframe['ptk_sig_w'],
                                  dataframe['pkj_sig_w'], dataframe['tkkj_sig_w'], dataframe['sasb_sig_w'], dataframe['ckh_sig_w']
                                  ])

        #TRIGGER
        dataframe['pic_l_s_trig'] =  3.9
        #print(dataframe['bs_pic_sig'])

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['runmode'].value in ('backtest', 'hyperopt'):
            assert (timeframe_to_minutes(self.timeframe) <= 5), "Backtest this strategy in 5m or 1m timeframe. Read comments for details."

        if self.timeframe == self.informative_timeframe:
            dataframe = self.do_indicators(dataframe, metadata)
        else:
            if not self.dp:
                return dataframe

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

            informative = self.do_indicators(informative.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
            # don't overwrite the base dataframe's HLCV information
           # skip_columns = [(s + "_" + self.informative_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
           # dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s, inplace=True)
        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[
                (dataframe['bs_pic_sig'] > dataframe['pic_l_s_trig'])
        ,
        'buy'] = 1


        # ERROR / DEBUG data
        print(dataframe['tenkan_sen_1h'])
        print(dataframe['tenkan_sen'])

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
                    (dataframe['bs_pic_sig'] < dataframe['pic_l_s_trig'])
        ,
        'sell'] = 1

        return dataframe
