import pandas as pd

from typing import List

from rolling_ta.indicator import Indicator

class RSI(Indicator):
    
    _gains = 0.0
    _losses = 0.0
    _prev_price = None
    _avg_gain = None
    _avg_loss = None
    _rsi = None
    _count = 0
    
    def __init__(self, series: pd.Series | list[float], period: int) -> None:
        """Rolling RSI Indicator

        Args:
            series (pd.Series): _description_
            period (int): _description_
        """
        super().__init__(period)
        
        
    def 
        
    def update(self, price: float):
        
        if self._prev_price is None:
            self._prev_price = price
        
        
    