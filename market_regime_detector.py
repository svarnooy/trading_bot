# bot/market_regime_detector.py
import numpy as np
import pandas as pd

def detect_market_regime(df: pd.DataFrame) -> str:
    """Определяет рыночный режим на основе 4H данных"""
    if len(df) < 50:
        return "trend"
        
    close = df['close'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    volume = df['volume'].to_numpy()
    
    # ATR за последние 14 свечей
    tr = np.maximum(high[1:] - low[1:], 
                   np.abs(high[1:] - close[:-1]),
                   np.abs(low[1:] - close[:-1]))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    atr_pct = atr / close[-1] if close[-1] > 0 else 0
    
    # Объём
    avg_vol = np.mean(volume[-20:])
    recent_vol = np.mean(volume[-5:])
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
    
    # EMA
    ema21 = pd.Series(close).ewm(span=21).mean().iloc[-1]
    ema55 = pd.Series(close).ewm(span=55).mean().iloc[-1]
    ema_distance = abs(ema21 - ema55) / ema55
    
    # Логика определения режима (более консервативная)
    if atr_pct > 0.04 and vol_ratio > 1.5 and ema_distance > 0.02:
        return "breakout"
    elif atr_pct < 0.01 and vol_ratio < 0.8 and ema_distance < 0.005:
        return "range"
    elif ema_distance > 0.01:
        return "trend"
    else:
        return "trend"  # по умолчанию — тренд (безопаснее)