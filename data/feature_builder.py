import numpy as np
import talib
from typing import List, Dict, Any

def _safe(arr):
    return np.array(arr, dtype=float)

def build_features_from_klines(klines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    klines: list of dict [{'open':...,'high':...,'low':...,'close':...,'volume':...}, ...]
    Returns feature dict with 'ok' flag. On failure returns {'ok': False, 'reason': ...}
    """
    if not klines or len(klines) < 30:
        return {"ok": False, "reason": "insufficient_bars"}

    try:
        close = _safe([k['close'] for k in klines])
        high  = _safe([k['high'] for k in klines])
        low   = _safe([k['low'] for k in klines])
        vol   = _safe([k['volume'] for k in klines])

        rsi = talib.RSI(close, 14)[-1]
        ema20 = talib.EMA(close, 20)[-1]
        ema50 = talib.EMA(close, 50)[-1]
        atr14 = talib.ATR(high, low, close, 14)[-1]

        macd, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
        macd_dir = 1 if macd[-1] > macd_signal[-1] else 0

        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        span = (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) != 0 else 1
        bb_pos = (close[-1] - lower[-1]) / span * 100

        avg_vol20 = vol[-20:].mean() if len(vol) >= 20 else vol.mean()
        volume_ratio = (vol[-1] / avg_vol20) if avg_vol20 > 0 else 0

        ema_diff = ((close[-1] - ema20) / ema20 * 100) if ema20 else 0
        trend_strength = ema_diff * macd_dir

        return {
            "ok": True,
            "rsi": float(rsi),
            "ema20": float(ema20),
            "ema50": float(ema50),
            "atr": float(atr14),
            "atr_percent": float((atr14 / close[-1]) * 100 if close[-1] > 0 else 0),
            "macd_direction": int(macd_dir),
            "macd_hist": float(macd_hist[-1]),
            "bb_position": float(bb_pos),
            "volume_ratio": float(volume_ratio),
            "ema_diff": float(ema_diff),
            "trend_strength": float(trend_strength),
            "close": float(close[-1]),
            "raw_volume": float(vol[-1])
        }
    except Exception as e:
        return {"ok": False, "reason": f"exception:{e}"}
