# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import logging
import datetime
from freqtrade.persistence import Trade
import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, stoploss_from_open
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# --------------------------------
def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class BBRSITV(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "ewo_high": 4.86,
        "for_ma_length": 22,
        "for_sigma": 1.74
    }

    # Sell hyperspace params:
    sell_params = {
        "for_ma_length_sell": 65,
        "for_sigma_sell": 1.895,
        "rsi_high": 72 
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.10  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.025  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False
    process_only_new_candles = True
    startup_candle_count = 30

    protections = [
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 1,
            "stop_duration": 60,
            "required_profit": -0.05
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 24,
            "trade_limit": 1,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.2
        },
    ]

    ewo_high = DecimalParameter(0, 7.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    for_sigma = DecimalParameter(0, 10.0, default=buy_params['for_sigma'], space='buy', optimize=True)
    for_sigma_sell = DecimalParameter(0, 10.0, default=sell_params['for_sigma_sell'], space='sell', optimize=True)
    rsi_high = IntParameter(60, 100, default=sell_params['rsi_high'], space='sell', optimize=True)
    for_ma_length = IntParameter(5, 80, default=buy_params['for_ma_length'], space='buy', optimize=True)
    for_ma_length_sell = IntParameter(5, 80, default=sell_params['for_ma_length_sell'], space='sell', optimize=True)

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Protection
    fast_ewo = 50
    slow_ewo = 200
 

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        src = 'close'
        for_rsi = 14
        dataframe['rsi'] = ta.RSI(dataframe[src], for_rsi)
        dataframe['rsi_4'] = ta.RSI(dataframe[src], 4)
        if self.config['runmode'].value == 'hyperopt':
            for for_ma in range(5, 81):
                dataframe[f'basis_{for_ma}'] = ta.EMA(dataframe['rsi'], for_ma)
                dataframe[f'dev_{for_ma}'] = ta.STDDEV(dataframe['rsi'], for_ma)
        else:
            dataframe[f'basis_{self.for_ma_length.value}'] = ta.EMA(dataframe['rsi'], self.for_ma_length.value)
            dataframe[f'basis_{self.for_ma_length_sell.value}'] = ta.EMA(dataframe['rsi'], self.for_ma_length_sell.value)
            dataframe[f'dev_{self.for_ma_length.value}'] = ta.STDDEV(dataframe['rsi'], self.for_ma_length.value)
            dataframe[f'dev_{self.for_ma_length_sell.value}'] = ta.STDDEV(dataframe['rsi'], self.for_ma_length_sell.value)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # variable trailing buy offset
        dataframe['perc'] = ((dataframe['high'].rolling(5).max()-dataframe['low'].rolling(5).min())/dataframe['low'].rolling(5).min()*100)
        dataframe['perc_norm'] = 2*((dataframe['perc'] - dataframe['perc'].rolling(50).min()) / (dataframe['perc'].rolling(50).max() - dataframe['perc'].rolling(50).min()))-1
       
        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < (dataframe[f'basis_{self.for_ma_length.value}'] - (dataframe[f'dev_{self.for_ma_length.value}'] * self.for_sigma.value))) &
                (dataframe['EWO'] >  self.ewo_high.value) &
                (dataframe['volume'] > 0)

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['rsi'] > self.rsi_high.value) |
                    (dataframe['rsi'] > dataframe[f'basis_{self.for_ma_length_sell.value}'] + ((dataframe[f'dev_{self.for_ma_length_sell.value}'] * self.for_sigma_sell.value)))
                ) &
                (dataframe['volume'] > 0)

            ),
            'sell'] = 1
        return dataframe

class BBRSITV5(BBRSITV):
    minimal_roi = {
        "0": 0.06
    }
    ignore_roi_if_buy_signal = True
    startup_candle_count = 400
    use_custom_stoploss = True

    stoploss = -0.10  # value loaded from strategy
    sell_params = {
        ##
        "pHSL": -0.10,
        "pPF_1": 0.01,
        "pPF_2": 0.048,
        "pSL_1": 0.009,
        "pSL_2": 0.043,
    }
    
    is_optimize_trailing = True
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < (dataframe[f'basis_{self.for_ma_length.value}'] - (dataframe[f'dev_{self.for_ma_length.value}'] * self.for_sigma.value)))
                &
                (
                    (
                        (dataframe['EWO'] > self.ewo_high.value)&
                        (dataframe['EWO'] < 10)
                    )
                    |
                    (
                        (dataframe['EWO'] >= 10)&
                        (dataframe['rsi'] < 40)
                    )
                )
                &
                (dataframe['rsi_4'] < 25)&
                (dataframe['volume'] > 0)
                # &
                # (dataframe["roc_bbwidth_max"] < 70)
            ),
            'buy'] = 1

        return dataframe

    pass

class BBRSITV5_TSL3D(BBRSITV5):
    # Orignal idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #
    # if process_only_new_candles = True, then you need to use 1m timeframe (and normal strategy timeframe as informative)
    # if process_only_new_candles = False, it will use ticker data and you won't need to change anything

    process_only_new_candles = False

    custom_info_trail_buy = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 300

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.1  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.002  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]:
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_dict
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def buy(self, dataframe, pair: str, current_price: float, buy_tag: str):
        dataframe.iloc[-1, dataframe.columns.get_loc('buy')] = 1
        ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
        if 'buy_tag' in dataframe.columns:
            dataframe.iloc[-1, dataframe.columns.get_loc('buy_tag')] = f"{buy_tag} ({ratio} %)"
        self.trailing_buy_info(pair, current_price)
        logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt  = abs((last_candle['perc_norm']))    #NOTE: Uzirox variable offset
        #default_offset = 0.005 
        default_offset = adapt*0.01

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if current_trailing_profit_ratio > 0 and last_candle['pre_buy'] == 1:
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            self.trailing_buy_info(pair, current_rate)
            self.trailing_buy(pair, reinit=True)
            logger.info(f'STOP trailing buy for {pair} because of {tag}')
        return tag

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        val = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)
        self.trailing_buy(pair, reinit=True)
        return val

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        # stop trailing when buy signal ! prevent from buying much higher price when slot is free
        self.trailing_buy_info(pair, rate)
        self.trailing_buy(pair, reinit=True)
        logger.info(f'STOP trailing buy for {pair} because I buy it')
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)

        if not self.trailing_buy_order_enabled or not self.config['runmode'].value in ('live', 'dry_run'): # no buy trailing
            return dataframe

        dataframe = dataframe.rename(columns={"buy": "pre_buy"})
        last_candle = dataframe.iloc[-1].squeeze()
        dataframe['buy'] = 0
        trailing_buy = self.trailing_buy(metadata['pair'])

        if not trailing_buy['trailing_buy_order_started'] and last_candle['pre_buy'] == 1:
            current_price = self.get_current_price(metadata["pair"], last_candle)
            open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
            if not open_trades:
                # start trailing buy
                self.custom_info_trail_buy[metadata["pair"]]['trailing_buy'] = {
                    'trailing_buy_order_started': True,
                    'trailing_buy_order_uplimit': last_candle['close'],
                    'start_trailing_price': last_candle['close'],
                    'buy_tag': last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal',
                    'start_trailing_time': datetime.now(timezone.utc),
                    'offset': 0,
                }
                self.trailing_buy_info(metadata["pair"], current_price)
                logger.info(f'start trailing buy for {metadata["pair"]} at {last_candle["close"]}')
        elif trailing_buy['trailing_buy_order_started']:
            current_price = self.get_current_price(metadata["pair"], last_candle)
            trailing_buy_offset = self.trailing_buy_offset(dataframe, metadata['pair'], current_price)

            if trailing_buy_offset == 'forcebuy':
                # buy in custom conditions
                self.buy(dataframe, metadata['pair'], current_price, trailing_buy['buy_tag'])
            elif trailing_buy_offset is None:
                # stop trailing buy custom conditions
                self.trailing_buy(metadata['pair'], reinit=True)
                logger.info(f'STOP trailing buy for {metadata["pair"]} because "trailing buy offset" returned None')
            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                # update uplimit
                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                self.custom_info_trail_buy[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'])
                self.custom_info_trail_buy[metadata["pair"]]['trailing_buy']['offset'] = trailing_buy_offset
                self.trailing_buy_info(metadata["pair"], current_price)
                logger.info(f'update trailing buy for {metadata["pair"]} at {old_uplimit} -> {self.custom_info_trail_buy[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
            elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                # buy ! current price > uplimit && lower thant starting price
                self.buy(dataframe, metadata['pair'], current_price, trailing_buy['buy_tag'])
            elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                # stop trailing buy because price is too high
                self.trailing_buy(metadata['pair'], reinit=True)
                self.trailing_buy_info(metadata["pair"], current_price)
                logger.info(f'STOP trailing buy for {metadata["pair"]} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
            else:
                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                self.trailing_buy_info(metadata["pair"], current_price)
                logger.info(f'price too high for {metadata["pair"]} !')
        return dataframe

    def get_current_price(self, pair: str, last_candle) -> float:
        if self.process_only_new_candles:
            current_price = last_candle['close']
        else:
            ticker = self.dp.ticker(pair)
            current_price = ticker['last']
        return current_price