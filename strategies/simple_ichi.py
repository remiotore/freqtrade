# Mainframe_Strat
# Obj: curva estable
#
#   entradas: para cada una definir: stake, stoploss, leverage, target exit (TE), emergency exit (EE), dca (entry conditions, quantity, amount).
#   FTP
#   Ichimoku
#   ct_morning
#   Gran Bollinger
#   VWAP

#   Secuencia de tareas
##   Importar bloques: custom_stake_amount, dca, custom_exit, custom_stoploss, custom_entry_rate, confirm_trade_exit 
#   Set up: 
#   Importar: Señal Ichimoku, probar en varios tf


################################################################################################################################
############################ IMPORTS ###########################################################################################
################################################################################################################################

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone, timedelta
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, informative, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.exchange import  timeframe_to_minutes #,timeframe_to_prev_date

import logging
logger = logging.getLogger(__name__)

################################################################################################################################
############################## CLASS DEFINITION ################################################################################
################################################################################################################################

class Simple_ichi(IStrategy):

################################################################################################################################
######## BASE VARIABLES ########################################################################################################

    use_exit_signal = True                      # Use exit signals produced by the strategy in addition to the minimal_roi.
    exit_profit_only = True                     # Do not exit if profit is less than offset
    exit_profit_offset = 0.0                    # Exit offset
    ignore_roi_if_entry_signal = False          # Do not exit if the entry signal is still active. This setting takes preference over minimal_roi and use_exit_signal.
    can_short = True                            # Futures
    startup_candle_count: int = 200             # 
    timeframe = '5m'                            #

################################################################################################################################
################################ ROI ###########################################################################################
################################################################################################################################
    minimal_roi = {
        "0": 10.0,     # 1000% (Disabled)
    }

    use_custom_roi = False
    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:
        
        """
        Custom ROI logic, returns a new minimum ROI threshold (as a ratio, e.g., 0.05 for +5%).
        Only called when use_custom_roi is set to True.

        If used at the same time as minimal_roi, an exit will be triggered when the lower
        threshold is reached. Example: If minimal_roi = {"0": 0.01} and custom_roi returns 0.05,
        an exit will be triggered if profit reaches 5%.

        :param pair: Pair that's currently analyzed.
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime.
        :param trade_duration: Current trade duration in minutes.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the current trade.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New ROI value as a ratio, or None to fall back to minimal_roi logic.
        """
                
        return None
################################################################################################################################
######################################## HYPEROPTABLES #########################################################################
################################################################################################################################
    
    # buyspace - Signal types
    optimize_buy_signals = False
    use_ichimoku_longs = BooleanParameter(default=True, space="buy", optimize=optimize_buy_signals)
    use_ichimoku_shorts = BooleanParameter(default=True, space="buy", optimize=optimize_buy_signals)
    ichimoku_timeframe = CategoricalParameter(['5m','15m','30m','1h','4h','1d'], default='4h', space="buy", optimize=optimize_buy_signals)

    # sellspace - Signal Types
    optimize_sell_signals = False
    use_ichimoku_exit_longs = BooleanParameter(default=True, space="sell", optimize=optimize_sell_signals)
    use_ichimoku_exit_shorts = BooleanParameter(default=True, space="sell", optimize=optimize_sell_signals)
################################################################################################################################
################################ PROTECTIONS ###################################################################################
################################################################################################################################
    @property
    def protections(self):
        """
            Defines the protections to apply during trading operations.
        """
        prot = []

        """  
        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value 
        })"""
        if True: #self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": 3*12, #self.stop_duration.value,
                "only_per_pair": True
            })

        return prot
################################################################################################################################################################################################################################################################
#                                STRATEGY CALLBACKS
################################################################################################################################################################################################################################################################

################################################################################################################################
################################ LEVERAGE ######################################################################################
################################################################################################################################
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
    #   """
    #   Customize leverage for each new trade. This method is only called in futures mode.
	#
    #    :param pair: Pair that's currently analyzed
    #    :param current_time: datetime object, containing the current datetime
    #    :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
    #    :param proposed_leverage: A leverage proposed by the bot.
    #    :param max_leverage: Max leverage allowed on this pair
    #    :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
    #    :param side: "long" or "short" - indicating the direction of the proposed trade
    #    :return: A leverage amount, which is between 1.0 and max_leverage.
    #    """
        return 1.0
################################################################################################################################
################################ CUSTOM STAKE AMOUNT ###########################################################################
################################################################################################################################   
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
            Calculates the stake amount to use for a trade, adjusted dynamically based on the DCA multiplier.
            - If the adjusted stake is lower than the allowed minimum (`min_stake`), it is automatically increased
              to meet the minimum stake requirement.
        """
        # Automatically adjusts to the minimum stake if it is too low.
        adjusted_stake = proposed_stake
        if proposed_stake < min_stake:
            adjusted_stake = min_stake

        return adjusted_stake
################################################################################################################################
################################ CUSTOM EXIT LOGIC #############################################################################
################################################################################################################################

################################################################################################################################
################################ CUSTOM STOPLOSS LOGIC #########################################################################
################################################################################################################################
    stoploss = -0.02                     #hard SL base
    use_custom_stoploss = False
    trailing_stop = False
################################################################################################################################
################################ CUSTOM PRICING LOGIC ##########################################################################
################################################################################################################################
    """def custom_entry_price(self, pair: str, trade: Trade | None, current_time: datetime, proposed_rate: float,
                           entry_tag: str | None, side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        return #new_entryprice

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, exit_tag: str | None, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)


        return #new_exitprice"""
################################################################################################################################
################################ CUSTOM TIMEOUT LOGIC ##########################################################################
################################################################################################################################
    """unfilledtimeout = {      #  Set unfilledtimeout to 25 hours, since the maximum timeout from below is 24 hours.
        "entry": 60 * 25,
        "exit": 60 * 25
    }

    def check_entry_timeout(self, pair: str, trade: Trade, order: Order,
                            current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob["bids"][0][0]
        # Cancel buy order if price is more than 2% above the order.
        if current_price > order.price * 1.02:
            return True
        return False


    def check_exit_timeout(self, pair: str, trade: Trade, order: Order,
                           current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob["asks"][0][0]
        # Cancel sell order if price is more than 2% below the order.
        if current_price < order.price * 0.98:
            return True
        return False"""
################################################################################################################################
################################ CONFIRM TRADE ENTRY ###########################################################################
################################################################################################################################
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders 
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        return True
################################################################################################################################
################################ CONFIRM TRADE EXIT ############################################################################
################################################################################################################################
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ["roi", "stop_loss", "stoploss_on_exchange", "trailing_stop_loss",
                           "exit_signal", "force_exit", "emergency_exit"]
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        if exit_reason == "force_exit" and trade.calc_profit_ratio(rate) < 0:
            # Reject force-sells with negative profit
            # This is just a sample, please adjust to your needs
            # (this does not necessarily make sense, assuming you know when you're force-selling)
            return False
        return True
################################################################################################################################
################################ ADJST TRADE POSITION (DCA) ####################################################################
################################################################################################################################
    
################################################################################################################################
################################ ADJST ORDER PRICE (ENTRY/EXIT) ################################################################
################################################################################################################################
    """def adjust_order_price(
        self,
        trade: Trade,
        order: Order | None,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: str | None,
        side: str,
        is_entry: bool,
        **kwargs,
    ) -> float | None:
        """
    """
        Exit and entry order price re-adjustment logic, returning the user desired limit price.
        This only executes when a order was already placed, still open (unfilled fully or partially)
        and not timed out on subsequent candles after entry trigger.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-callbacks/

        When not implemented by a strategy, returns current_order_rate as default.
        If current_order_rate is returned then the existing order is maintained.
        If None is returned then order gets canceled but not replaced by a new one.

        :param pair: Pair that's currently analyzed
        :param trade: Trade object.
        :param order: Order object
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate, calculated based on pricing settings in entry_pricing.
        :param current_order_rate: Rate of the existing order in place.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param is_entry: True if the order is an entry order, False if it's an exit order.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float or None: New entry price value if provided"""
    """

        # Limit entry orders to use and follow SMA200 as price target for the first 10 minutes since entry trigger for BTC/USDT pair.
        if (
            is_entry
            and pair == "BTC/USDT" 
            and entry_tag == "long_sma200" 
            and side == "long" 
            and (current_time - timedelta(minutes=10)) <= trade.open_date_utc
        ):
            # just cancel the order if it has been filled more than half of the amount
            if order.filled > order.remaining:
                return None
            else:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
                current_candle = dataframe.iloc[-1].squeeze()
                # desired price
                return current_candle["sma_200"]
        # default: maintain existing order
        return current_order_rate
    """
################################################################################################################################
################################ ORDER FILLED (CALLED WHEN AN ORDER FILLS) #####################################################
################################################################################################################################
    """def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
    """Called right after an order fills. 
        Will be called for all order types (entry, exit, stoploss, position adjustment).
        :param pair: Pair for trade
        :param trade: trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """"""
        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if (trade.nr_of_successful_entries == 1) and (order.ft_order_side == trade.entry_side):
            trade.set_custom_data(key="entry_candle_high", value=last_candle["high"])

        return None"""
################################################################################################################################
################################ CHART ANNOTTATIONS ############################################################################
################################################################################################################################

################################################################################################################################
######################################## PLOT CONFIGURATION ####################################################################
################################################################################################################################
    plot_config = {
        'main_plot': {

                #'sma200_1h': {'color': 'white','plotly': {'opacity': 0.9}},

                #'tenkan_sen_1h': {'color': 'blue', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'kijun_sen_1h': {'color': 'red', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'senkou_span_a_1h': {'color': 'green', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'none', 'dash': 'dash'}},
                #'senkou_span_b_1h': {'color': 'brown', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'senkou_span_a_1h', 'dash': 'dash'}},
                #'chikou_span_1h': {'color': 'gray', 'plotly': {'opacity': 0.9, 'width': 1.0, 'dash': 'dot'}},

                #'tenkan_sen': {'color': 'blue', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'kijun_sen': {'color': 'red', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'senkou_span_a': {'color': 'green', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'none', 'dash': 'dash'}},
                #'senkou_span_b': {'color': 'brown', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'senkou_span_a_1h', 'dash': 'dash'}},
                #'chikou_span': {'color': 'gray', 'plotly': {'opacity': 0.9, 'width': 1.0, 'dash': 'dot'}},     

                #'tenkan_sen_1d': {'color': 'blue', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'kijun_sen_1d': {'color': 'red', 'plotly': {'opacity': 1.0, 'width': 1.5}},
                #'senkou_span_a_1d': {'color': 'green', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'none', 'dash': 'dash'}},
                #'senkou_span_b_1d': {'color': 'brown', 'plotly': {'opacity': 0.0, 'width': 1.0, 'fill': 'senkou_span_a_1h', 'dash': 'dash'}},
                #'chikou_span_1d': {'color': 'gray', 'plotly': {'opacity': 0.9, 'width': 1.0, 'dash': 'dot'}},     
                
                #'BB_upper_1h': {'color': 'yellow','plotly': {'opacity': 0.9}},
                #'BB_lower_1h': {'color': 'yellow','plotly': {'opacity': 0.9}},
                #'BB_middle_1h': {'color': 'yellow','plotly': {'opacity': 0.9}},
                #'BB_upper_SL_1h': {'color': 'red','plotly': {'opacity': 0.9}},
                #'BB_lower_SL_1h': {'color': 'red','plotly': {'opacity': 0.9}},

                                       

        },
        'subplots': {
   
            
            "ADX": {
                #'adx_1h': {'color': 'yellow'},
                #'pdi_1h': {'color': 'green', 'fill_to': 'adx'},
                #'mdi_1h': {'color': 'red', 'fill_to': 'adx'}
            },
            "ADX_CROSSDIR": {
                #'adx_cross_dir': {'color': 'yellow'}
            },

            "RSI": {
               # 'rsi_1h': {'color': 'rgba(94, 43, 126, 0.87)'},
            },
            
            "BTC 1h": {
               # 'btc_usdt_close_1h': {'color': 'orange'},
               # 'btc_usdt_sma200_1h': {'color': 'white','plotly': {'opacity': 0.9}}
            },
        },
    }
################################################################################################################################################################################################################################################################
#                                        POPULATOR METHODS
################################################################################################################################################################################################################################################################

################################################################################################################################
######################################## POPULATE INDICATORS (informative) #####################################################
################################################################################################################################
    @informative('4h')
    def populate_indicators_4h(self, df: DataFrame, metadata: dict) -> DataFrame:
        # ICHIMOKU CLOUD ################################################################################################################################################################
        if self.ichimoku_timeframe.value == '4h':
            high = df['high']; low = df['low']; close = df['close']
            df['tenkan_sen'] = (high.rolling(9).max() + low.rolling(9).min()) / 2                        # rapida
            df['kijun_sen'] = (high.rolling(26).max() + low.rolling(26).min()) / 2                       # lenta
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)                   # nube: lado alcisa
            df['senkou_span_b'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)       # nube: lado bajista
            df['chikou_span'] = close.shift(-26)                                                         # lagged: referencia (no usar en señales, mete lookahead-bias)

        # OTROS
            df['mfi'] = ta.MFI(df, timeperiod=14)

        return df
################################################################################################################################
######################################## POPULATE INDICATORS (self.timeframe) ##################################################
################################################################################################################################
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
            Calculates technical indicators used to define entry and exit signals.
        """
        return df
################################################################################################################################
######################################## POPULATE ENTRY TREND ##################################################################
################################################################################################################################
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        # ICHIMOKU #############################################################################################################
        ichi_tf = self.ichimoku_timeframe.value
        if self.use_ichimoku_longs.value:                                                   #Flag set in Hyperoptables
            ichimoku_long = (
            (df[f'close_{ichi_tf}'] > df[f'senkou_span_a_{ichi_tf}'])                                       #por arriba de nube
            & (df[f'senkou_span_a_{ichi_tf}'] > df[f'senkou_span_b_{ichi_tf}'])                                   #nube alcista
            & qtpylib.crossed_above(df[f'tenkan_sen_{ichi_tf}'], df[f'kijun_sen_{ichi_tf}']).astype(int)    #rapida cruza sobre lenta
            #& (df[f'mfi_{ichi_tf}'] < 35)                                                                                         #filtro adicional: mfi "barato"
        )

            df.loc[
                (
                    (ichimoku_long)                                    #ichimoku mask
                ),
                ['enter_long', 'enter_tag']
            ] = (1, 'ރIchimoku Long')

        if self.use_ichimoku_shorts.value:                                                   #Flag set in Hyperoptables
            ichimoku_short = (
            (df[f'close_{ichi_tf}'] < df[f'senkou_span_b_{ichi_tf}']) &                                           #por debajo de nube
            (df[f'senkou_span_b_{ichi_tf}'] < df[f'senkou_span_a_{ichi_tf}']) &                                   #nube bajista
            qtpylib.crossed_above(df[f'kijun_sen_{ichi_tf}'], df[f'tenkan_sen_{ichi_tf}']).astype(int)           #lenta sobre rápida
            #& (df[f'mfi_{ichi_tf}'] > 65)                                                                                                 #filtro adicional: mfi "caro"
        )
            
            df.loc[
                (
                    (ichimoku_short)                                       #ichimoku mask
                ),
                ['enter_short', 'enter_tag']
            ] = (1, 'ރIchimoku Short')

        return df
################################################################################################################################
######################################## POPULATE EXIT TREND ###################################################################
################################################################################################################################
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # ICHIMOKU EXIT ########################################################################################################
        ichi_tf = self.ichimoku_timeframe.value
        if self.use_ichimoku_exit_longs.value:                                                           #Flag set in Hyperoptables
            ichimoku_keep_long = (
            (df[f'close_{ichi_tf}'] > df[f'senkou_span_a_{ichi_tf}'])                                       #por arriba de nube
            & (df[f'senkou_span_a_{ichi_tf}'] > df[f'senkou_span_b_{ichi_tf}'])                                   #nube alcista
        )

            df.loc[
                (
                    ~(ichimoku_keep_long)                                    #ichimoku mask
                ),
                ['exit_long', 'exit_tag']
            ] = (1, 'SEL Ichimoku Broken')

        if self.use_ichimoku_exit_shorts.value:                                                   #Flag set in Hyperoptables
            ichimoku_keep_short = (
            (df[f'close_{ichi_tf}'] < df[f'senkou_span_b_{ichi_tf}'])                                              #por debajo de nube
            & (df[f'senkou_span_b_{ichi_tf}'] < df[f'senkou_span_a_{ichi_tf}'])                                    #nube bajista
        )
            
            df.loc[
                (
                    ~(ichimoku_keep_short)                                       #ichimoku mask
                ),
                ['exit_short', 'exit_tag']
            ] = (1, 'SES Ichimoku Broken')
            
            return df