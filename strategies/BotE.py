

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from sqlalchemy import create_engine
import sqlite3


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import os
import pathlib
import time
from sqlite3 import Error
from keras.models import load_model



class BotE(IStrategy):

    INTERFACE_VERSION = 2
    minimal_roi = {
        "0": 0.0739,
        "29": 0.0572,
        "62": 0.01108,
        "84": 0




    }


    stoploss = -1.10

    trailing_stop = True
    trailing_stop_positive = 0.02543
    trailing_stop_positive_offset = 0.08718
    trailing_only_offset_is_reached = True





    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {

        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {

            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)


        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]
        dataframe["kc_percent"] = (
             (dataframe["close"] - dataframe["kc_lowerband"]) /
             (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        )
        dataframe["kc_width"] = (
             (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        )

        dataframe['uo'] = ta.ULTOSC(dataframe)

        dataframe['cci'] = ta.CCI(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']



        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['roc'] = ta.ROC(dataframe)



        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        weighted_bollinger = qtpylib.weighted_bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        dataframe["wbb_percent"] = (
            (dataframe["close"] - dataframe["wbb_lowerband"]) /
            (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        )
        dataframe["wbb_width"] = (
            (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) / dataframe["wbb_middleband"]
        )



        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema150'] = ta.EMA(dataframe, timeperiod=150)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema250'] = ta.EMA(dataframe, timeperiod=250)


        dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['sma150'] = ta.SMA(dataframe, timeperiod=150)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma250'] = ta.SMA(dataframe, timeperiod=250)

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)



        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']



        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)

        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)

        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)

        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]

        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]

        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]



        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)

        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)

        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)

        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)

        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)

        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)



        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)

        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]

        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]

        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]

        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]

        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]



        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        print(str(metadata))
        Table=str(metadata['pair'])
        print(Table)
        Table = Table.replace("/" , "_")
        engine = create_engine('sqlite:///Neural.sqlite', echo=True)
        sqlite_connection = engine.connect()
        engine.execute("DROP TABLE IF EXISTS "+Table)
        sqlite_table = Table
        normed_df = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
        normed_df['date'] = dataframe['date'].values
        normed_df['open'] = dataframe['open'].values
        normed_df.to_sql(sqlite_table, sqlite_connection, if_exists='fail')
        sqlite_connection.close()
        print("database closed")
        NeuralDf=normed_df.drop(columns=['date', 'open'])

        df2 = NeuralDf
        WEIGHTS = Table + "Weight"
        files0 = pathlib.Path("/tmp/"+WEIGHTS+"_model.h5")
        if os.path.exists("/tmp/" + WEIGHTS + "_wait"):
            time.sleep(11)
            os.remove("/tmp/" + WEIGHTS + "_wait")
        if files0.exists():
            model = load_model(files0)
            files0 = pathlib.Path("/tmp/"+WEIGHTS+".h5")
            if files0.exists ():
                model.load_weights(files0)
            train = df2.replace(np.nan, 0.0)
            l2 = (model.predict(train).round())
            column_values = ['b', 's']
            df5 = pd.DataFrame(data=l2, columns=column_values)
            dataframe['s'] = df5['s']
            dataframe['b'] = df5['b']
        else:
            dataframe['s'] = 0.0
            dataframe['b'] = 0.0
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
               (dataframe['b'] > 0.7 ) &
               (dataframe['s'] < 0.3) &







                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                 (dataframe['s'] > 0.7) &
                 (dataframe['b'] < 0.3) &












                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
