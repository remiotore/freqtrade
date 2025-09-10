import numpy as np
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from stable_baselines3 import PPO


class GymStrategy_2(IStrategy):
    stoploss = -0.20
    trailing_stop = False
    ticker_interval = '5m'
    process_only_new_candles = False
    startup_candle_count: int = 20
    model = None

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._load_model()

    def _load_model(self):
        try:
            self.model = PPO.load('/freqtrade/user_data/model.gym')
        except FileNotFoundError:

            pass

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

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
































        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)


























































        """

        if self.dp:
            if self.dp.runmode in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        action, nan_list = self.rl_model_redict(dataframe)
        dataframe.loc[action == 1, 'buy'] = 1
        dataframe.loc[nan_list == True, 'buy'] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        action, nan_list = self.rl_model_redict(dataframe)
        dataframe.loc[action == 2, 'sell'] = 1
        dataframe.loc[nan_list == True, 'sell'] = 0
        return dataframe

    def rl_model_redict(self, dataframe):
        data = np.array([
            dataframe['adx'],
            dataframe['plus_dm'],
            dataframe['plus_di'],
            dataframe['minus_dm'],
            dataframe['minus_di'],
            dataframe['aroonup'],
            dataframe['aroondown'],
            dataframe['aroonosc'],
            dataframe['ao'],


            dataframe['uo'],
            dataframe['cci'],
            dataframe['rsi'],
            dataframe['fisher_rsi'],
            dataframe['slowd'],
            dataframe['slowk'],
            dataframe['fastd'],
            dataframe['fastk'],
            dataframe['fastd_rsi'],
            dataframe['fastk_rsi'],
            dataframe['macd'],
            dataframe['macdsignal'],
            dataframe['macdhist'],
            dataframe['mfi'],
            dataframe['roc'],


























        ], dtype=np.float)

        data = data.reshape(-1, 24)

        nan_list = np.isnan(data).any(axis=1)
        data = np.nan_to_num(data)
        action, _ = self.model.predict(data, deterministic=True)

        return action, nan_list
