# --- Imports required for ROI table
from freqtrade.optimize.space import SKDecimal
from skopt.space import Categorical, Dimension, Integer
import math

class SampleROITables(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.076,
        "8": 0.07,
        "17": 0.06,
        "20": 0.04,
        "31": 0.02,
        "45": 0.01,
        "56": 0.004,
        "79": 0
    }

    # Stoploss:
    stoploss = -0.125

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0033
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            """
            Create a ROI table.

            Generates the ROI table that will be used by Hyperopt.
            You may override it in your custom Hyperopt class.
            """
            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5'] + params['roi_p6'] + params['roi_p7'] + params['roi_p8']
            roi_table[params['roi_t8']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5'] + params['roi_p6'] + params['roi_p7']
            roi_table[params['roi_t8'] + params['roi_t7']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5'] + params['roi_p6']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5'] + params['roi_t4']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3']] = params['roi_p1'] + params['roi_p2']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2']] = params['roi_p1']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0
            
            return roi_table
        
        @staticmethod
        def roi_space() -> List[Dimension]:
            """
            Create a ROI space.

            Defines values to search for each ROI steps.

            This method implements adaptive roi hyperspace with varied
            ranges for parameters which automatically adapts to the
            timeframe used.

            It's used by Freqtrade by default, if no custom roi_space method is defined.
            """

            # Default scaling coefficients for the roi hyperspace. Can be changed
            # to adjust resulting ranges of the ROI tables.
            # Increase if you need wider ranges in the roi hyperspace, decrease if shorter
            # ranges are needed.
            roi_t_alpha = 1.0
            roi_p_alpha = 1.0

            timeframe_min = 1

            # We define here limits for the ROI space parameters automagically adapted to the
            # timeframe used by the bot:
            #
            # * 'roi_t' (limits for the time intervals in the ROI tables) components
            #   are scaled linearly.
            # * 'roi_p' (limits for the ROI value steps) components are scaled logarithmically.
            #
            # The scaling is designed so that it maps exactly to the legacy Freqtrade roi_space()
            # method for the 5m timeframe.
            roi_t_scale = timeframe_min / 5
            roi_p_scale = math.log1p(timeframe_min) / math.log1p(5)
            roi_limits = {
                'roi_t1_min': int(10 * roi_t_scale * roi_t_alpha),
                'roi_t1_max': int(120 * roi_t_scale * roi_t_alpha),
                'roi_t2_min': int(10 * roi_t_scale * roi_t_alpha),
                'roi_t2_max': int(105 * roi_t_scale * roi_t_alpha),
                'roi_t3_min': int(10 * roi_t_scale * roi_t_alpha),
                'roi_t3_max': int(90 * roi_t_scale * roi_t_alpha),
                'roi_t4_min': int(10 * roi_t_scale * roi_t_alpha),
                'roi_t4_max': int(75 * roi_t_scale * roi_t_alpha),
                'roi_t5_min': int(10 * roi_t_scale * roi_t_alpha),
                'roi_t5_max': int(60 * roi_t_scale * roi_t_alpha),
                'roi_t6_min': int(10 * roi_t_scale * roi_t_alpha),
                'roi_t6_max': int(45 * roi_t_scale * roi_t_alpha),
                'roi_t7_min': int(10 * roi_t_scale * roi_t_alpha),
                'roi_t7_max': int(35 * roi_t_scale * roi_t_alpha),
                'roi_t8_min': int(10 * roi_t_scale * roi_t_alpha),
                'roi_t8_max': int(30 * roi_t_scale * roi_t_alpha),
                'roi_p1_min': 0.01 * roi_p_scale * roi_p_alpha,
                'roi_p1_max': 0.04 * roi_p_scale * roi_p_alpha,
                'roi_p2_min': 0.01 * roi_p_scale * roi_p_alpha,
                'roi_p2_max': 0.045 * roi_p_scale * roi_p_alpha,
                'roi_p3_min': 0.01 * roi_p_scale * roi_p_alpha,
                'roi_p3_max': 0.055 * roi_p_scale * roi_p_alpha,
                'roi_p4_min': 0.01 * roi_p_scale * roi_p_alpha,
                'roi_p4_max': 0.065 * roi_p_scale * roi_p_alpha,
                'roi_p5_min': 0.01 * roi_p_scale * roi_p_alpha,
                'roi_p5_max': 0.7 * roi_p_scale * roi_p_alpha,
                'roi_p6_min': 0.01 * roi_p_scale * roi_p_alpha,
                'roi_p6_max': 0.10 * roi_p_scale * roi_p_alpha,
                'roi_p7_min': 0.01 * roi_p_scale * roi_p_alpha,
                'roi_p7_max': 0.15 * roi_p_scale * roi_p_alpha,
                'roi_p8_min': 0.01 * roi_p_scale * roi_p_alpha,
                'roi_p8_max': 0.20 * roi_p_scale * roi_p_alpha,
            }
            p = {
                'roi_t1': roi_limits['roi_t1_min'],
                'roi_t2': roi_limits['roi_t2_min'],
                'roi_t3': roi_limits['roi_t3_min'],
                'roi_t4': roi_limits['roi_t4_min'],
                'roi_t5': roi_limits['roi_t5_min'],
                'roi_t6': roi_limits['roi_t6_min'],
                'roi_t7': roi_limits['roi_t7_min'],
                'roi_t8': roi_limits['roi_t8_min'],
                'roi_p1': roi_limits['roi_p1_min'],
                'roi_p2': roi_limits['roi_p2_min'],
                'roi_p3': roi_limits['roi_p3_min'],
                'roi_p4': roi_limits['roi_p4_min'],
                'roi_p5': roi_limits['roi_p5_min'],
                'roi_p6': roi_limits['roi_p6_min'],
                'roi_p7': roi_limits['roi_p7_min'],
                'roi_p8': roi_limits['roi_p8_min'],
            }
            p = {
                'roi_t1': roi_limits['roi_t1_max'],
                'roi_t2': roi_limits['roi_t2_max'],
                'roi_t3': roi_limits['roi_t3_max'],
                'roi_t4': roi_limits['roi_t4_max'],
                'roi_t5': roi_limits['roi_t5_max'],
                'roi_t6': roi_limits['roi_t6_max'],
                'roi_t7': roi_limits['roi_t7_max'],
                'roi_t8': roi_limits['roi_t8_max'],
                'roi_p1': roi_limits['roi_p1_max'],
                'roi_p2': roi_limits['roi_p2_max'],
                'roi_p3': roi_limits['roi_p3_max'],
                'roi_p4': roi_limits['roi_p4_max'],
                'roi_p5': roi_limits['roi_p5_max'],
                'roi_p6': roi_limits['roi_p6_max'],
                'roi_p7': roi_limits['roi_p7_max'],
                'roi_p8': roi_limits['roi_p8_max'],
            }

            return [
                Integer(roi_limits['roi_t1_min'], roi_limits['roi_t1_max'], name='roi_t1'),
                Integer(roi_limits['roi_t2_min'], roi_limits['roi_t2_max'], name='roi_t2'),
                Integer(roi_limits['roi_t3_min'], roi_limits['roi_t3_max'], name='roi_t3'),
                Integer(roi_limits['roi_t4_min'], roi_limits['roi_t4_max'], name='roi_t4'),
                Integer(roi_limits['roi_t5_min'], roi_limits['roi_t5_max'], name='roi_t5'),
                Integer(roi_limits['roi_t6_min'], roi_limits['roi_t6_max'], name='roi_t6'),
                Integer(roi_limits['roi_t7_min'], roi_limits['roi_t7_max'], name='roi_t7'),
                Integer(roi_limits['roi_t8_min'], roi_limits['roi_t8_max'], name='roi_t8'),
                SKDecimal(roi_limits['roi_p1_min'], roi_limits['roi_p1_max'], decimals=3, name='roi_p1'),
                SKDecimal(roi_limits['roi_p2_min'], roi_limits['roi_p2_max'], decimals=3, name='roi_p2'),
                SKDecimal(roi_limits['roi_p3_min'], roi_limits['roi_p3_max'], decimals=3, name='roi_p3'),
                SKDecimal(roi_limits['roi_p4_min'], roi_limits['roi_p4_max'], decimals=3, name='roi_p4'),
                SKDecimal(roi_limits['roi_p5_min'], roi_limits['roi_p5_max'], decimals=3, name='roi_p5'),
                SKDecimal(roi_limits['roi_p6_min'], roi_limits['roi_p6_max'], decimals=3, name='roi_p6'),
                SKDecimal(roi_limits['roi_p7_min'], roi_limits['roi_p7_max'], decimals=3, name='roi_p7'),
                SKDecimal(roi_limits['roi_p8_min'], roi_limits['roi_p8_max'], decimals=3, name='roi_p8'),
            ]
