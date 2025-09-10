from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
import technical.indicators as ftt
import math
import logging

logger = logging.getLogger(__name__)

class BSS2b(IStrategy):
    INTERFACE_VERSION = 2
    
    buy_params = {
        "base_nb_candles_buy": 12,
        "rsi_buy": 58,
        "ewo_high": 3.001, #3.001
        "ewo_low": -10.289, #-10.289
        "low_offset": 0.987, #0.987
        "lambo2_ema_14_factor": 0.981, #0.981
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 39, #39
        "lambo2_rsi_4_limit": 44, #44
        "buy_adx": 20,
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179
    }

    sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.07,
        "high_offset_2": 1.05
    }


    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
            {"method": "MaxDrawdown", "lookback_period_candles": 48, "trade_limit": 20, "stop_duration_candles": 4, "max_allowed_drawdown": 0.2},
            {"method": "StoplossGuard", "lookback_period_candles": 24, "trade_limit": 4, "stop_duration_candles": 2, "only_per_pair": False},
            {"method": "LowProfitPairs", "lookback_period_candles": 6, "trade_limit": 2, "stop_duration_candles": 60, "required_profit": 0.02},
            {"method": "LowProfitPairs", "lookback_period_candles": 24, "trade_limit": 4, "stop_duration_candles": 2, "required_profit": 0.01}
        ]

    stoploss = -0.99

    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    lambo2_enabled = bool(buy_params['lambo2_enabled'])
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    fast_ewo = 60
    slow_ewo = 220
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 3.4, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=False)

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.0135
    trailing_only_offset_is_reached = True

    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, optimize=is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize=is_optimize_cofi)

    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}

    timeframe = '5m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 400

    plot_config = {'main_plot': {'ma_buy': {'color': 'orange'}, 'ma_sell': {'color': 'orange'}}}




    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        rsi_value = dataframe['rsi'].iloc[-1]
        ema_value = dataframe['ema_14'].iloc[-1]
        price = current_rate



        if (current_time - trade.open_date_utc).total_seconds() / 3600 >= 16:
            if current_profit < -0.10 and rsi_value > 30 and price < ema_value:  # Only exit if profit is less than -10%
                return 'market_based_exit'  # Exit reason (optional)
            else:
                return False  # Hold the trade (profitable even after 16 hours)


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDT', 'BUSD', 'USDC', 'DAI', 'TUSD', 'PAX', 'USD', 'EUR', 'GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs += [(btc_info_pair, self.timeframe), (btc_info_pair, self.inf_1h)]
    
        return informative_pairs

    def pump_dump_protection(self, dataframe, metadata):
        df36h = dataframe.copy().shift(432)
        df24h = dataframe.copy().shift(288)

        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()

        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])

        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()

        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)

        return dataframe

    def base_tf_btc_indicators(self, dataframe, metadata):
        dataframe['price_trend_long'] = (dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())

        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def info_tf_btc_indicators(self, dataframe, metadata):
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)

        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def EWO(self, dataframe, ema_length=5, ema2_length=3):
        ema1 = ta.EMA(dataframe, timeperiod=ema_length)
        ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
        emadif = (ema1 - ema2) / ema2 * 100
        return emadif


    def populate_indicators(self, dataframe, metadata):
        """
        Calcula os indicadores técnicos utilizados pela estratégia.

        Argumentos:
            dataframe (pandas.DataFrame): DataFrame com dados históricos do mercado.
            metadata (dict): Dicionário com metadados sobre o par e a timeframe.

        Retorna:
            pandas.DataFrame: DataFrame com os indicadores técnicos adicionados.
        """


        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['EWO'] = self.EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)

        dataframe = self.pump_dump_protection(dataframe, metadata)


        if self.config['stake_currency'] in ['USDT', 'BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        dataframe['ema_long_term'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['volume_avg'] = ta.SMA(dataframe['volume'], timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        lambo2 = (
            bool(self.lambo2_enabled) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2, 'buy_tag'] += 'lambo2_'
        conditions.append(lambo2)

        buy1ewo = (
            (dataframe['rsi_fast'] < 35) &
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
            (dataframe['EWO'] > self.ewo_high.value) &
            (dataframe['rsi'] < self.rsi_buy.value) &
            (dataframe['volume'] > 0) &
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy1ewo, 'buy_tag'] += 'buy1eworsi_'
        conditions.append(buy1ewo)

        buy2ewo = (
            (dataframe['rsi_fast'] < 35) &
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
            (dataframe['EWO'] < self.ewo_low.value) &
            (dataframe['volume'] > 0) &
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy2ewo, 'buy_tag'] += 'buy2ewo_'
        conditions.append(buy2ewo)

        is_cofi = (
            (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
            (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
            (dataframe['fastk'] < self.buy_fastk.value) &
            (dataframe['fastd'] < self.buy_fastd.value) &
            (dataframe['adx'] > self.buy_adx.value) &
            (dataframe['EWO'] > self.buy_ewo_high.value)
        )
        dataframe.loc[is_cofi, 'buy_tag'] += 'cofi_'
        conditions.append(is_cofi)

        is_entry_confirmed = (
            (dataframe['close'] > dataframe['ema_long_term']) &
            (dataframe['volume'] > dataframe['volume_avg'])
        )

        buy_signal = conditions[0]
        for condition in conditions[1:]:
            buy_signal = buy_signal | condition

        dataframe.loc[buy_signal, 'buy'] = 1
        return dataframe


    def populate_exit_trend(self, dataframe, metadata):
        conditions = []
        dataframe.loc[:, 'sell_tag'] = ''

        sell1 = (
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value))
        )
        dataframe.loc[sell1, 'sell_tag'] += 'sell1_'
        conditions.append(sell1)

        sell2 = (
            (dataframe['close'] > (dataframe['hma_50'] * 1.015)) &
            (dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0) &
            (dataframe['volume_change_percentage'] > 2.0) &
            (dataframe['rsi_mean'] > 50)
        )
        dataframe.loc[sell2, 'sell_tag'] += 'sell2_'
        conditions.append(sell2)

        dataframe['sell_tag'] = dataframe['sell_tag'].apply(lambda x: x[:-1] if x != '' else x)

        is_exit_confirmed = (
            (dataframe['close'] < dataframe['ema_long_term']) &                         # Preço abaixo da EMA de longo prazo
            (dataframe['volume'] < dataframe['volume_avg']) &                           # Volume abaixo da média móvel do volume
            (dataframe['close'].rolling(20).std() > 0.02) &                             # Desvio padrão dos últimos 20 períodos de fechamento maior que 2%
            (qtpylib.crossed_above(dataframe['rsi'], 70))                              # Cruzamento do preço abaixo do RSI
        )

        sell_signal = conditions[0]
        for condition in conditions[1:]:
            sell_signal = sell_signal | condition

        dataframe.loc[sell_signal, 'sell'] = 1
        return dataframe

