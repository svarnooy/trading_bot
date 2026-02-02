# bot/strategy_factory.py
from .strategies.trend import TrendStrategy
from .strategies.range import RangeStrategy
from .strategies.breakout import BreakoutStrategy
from .strategies.countertrend import CounterTrendStrategy
from .market_regime_detector import detect_market_regime

def create_strategy(cfg, df_1h):
    # Если отключена полная стратегия — только тренд
    if hasattr(cfg, 'use_full_strategy') and not cfg.use_full_strategy:
        return TrendStrategy(cfg)
    
    # Определяем режим автоматически
    regime = detect_market_regime(df_1h)
    
    if regime == "trend":
        return TrendStrategy(cfg)
    elif regime == "range":
        return RangeStrategy(cfg)
    elif regime == "breakout":
        return BreakoutStrategy(cfg)
    else:
        return CounterTrendStrategy(cfg)