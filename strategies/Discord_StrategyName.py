from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
import logging
from skopt.space import Dimension, Integer
from freqtrade.misc import round_dict

logger = logging.getLogger(__name__)

class StrategyName(IStrategy):
    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table = {}
            roi_table[params['roi_t1']] = 0
            return roi_table

        @staticmethod
        def roi_space() -> List[Dimension]:
            roi_min_time = 10
            roi_max_time = 600

            roi_limits = {
                'roi_t1_min': int(roi_min_time),
                'roi_t1_max': int(roi_max_time)
            }
            logger.debug(f"Using roi space limits: {roi_limits}")
            p = {
                'roi_t1': roi_limits['roi_t1_min']
            }
            logger.info(f"Min roi table: {round_dict(StrategyName.HyperOpt.generate_roi_table(p), 3)}")
            p = {
                'roi_t1': roi_limits['roi_t1_max']
            }
            logger.info(f"Max roi table: {round_dict(StrategyName.HyperOpt.generate_roi_table(p), 3)}")

            return [
                Integer(roi_limits['roi_t1_min'], roi_limits['roi_t1_max'], name='roi_t1')
            ]