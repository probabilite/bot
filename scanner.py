import asyncio
import json
import logging
import os
import time
import random
import warnings
import yaml
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, List, Optional, Tuple
import math
import aiohttp
import joblib
import numpy as np
import pandas as pd
import talib
from binance import AsyncClient
from dotenv import load_dotenv
from requests.exceptions import RequestException
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from html import escape as _escape
import re
from pathlib import Path
from adaptive_manager import AdaptiveParameterManager

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# ENV ve Global Ayarlar
# =========================
load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
POSITION_FILE = str(BASE_DIR / "positions.json")
HISTORY_FILE = str(BASE_DIR / "history_reinforced.json")
SYMBOL_CACHE_FILE = str(BASE_DIR / "symbol_cache.json")
BLACKLIST_FILE = str(BASE_DIR / "blacklist.json")
CONFIG_FILE = str(BASE_DIR / "config.yml")
RISK_STATE_FILE = str(BASE_DIR / "risk_state.json")

# --- Logging ---
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

binance_ws_logger = logging.getLogger("binance.ws.reconnecting_websocket")
binance_ws_logger.setLevel(logging.WARNING)
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.WARNING)

# --- Config / ML / Runtime Params ---
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 20))
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "15m")
SUPPORTED_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1M"
}

def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, "")
    if v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

_interval_warned = False
def get_valid_interval(iv: Optional[str]) -> str:
    global _interval_warned
    iv = (iv or "15m").strip()
    if iv not in SUPPORTED_INTERVALS:
        if not _interval_warned:
            logger.warning(f"Geçersiz interval '{iv}' tespit edildi. '15m' kullanılacak.")
            _interval_warned = True
        return "15m"
    return iv

MODEL_CLASSIFICATION_PATH = os.getenv("MODEL_CLASSIFICATION_PATH", "model_cls.pkl")
MODEL_REGRESSION_PATH = os.getenv("MODEL_REGRESSION_PATH", "model_reg.pkl")
USE_REGRESSION_FOR_SIZING = env_bool("USE_REGRESSION_FOR_SIZING", False)
FILTER_NEG_REGRESSION = env_bool("FILTER_NEG_REGRESSION", False)
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", 3))
TRAILING_OFFSET_ENV = float(os.getenv("TRAILING_OFFSET", 0.5))
DAILY_REPORT_TIME = os.getenv("DAILY_REPORT_TIME", "00:00")
AUTO_RETRAIN = env_bool("AUTO_RETRAIN", False)
DISABLE_STARTUP_TRAINING = env_bool("DISABLE_STARTUP_TRAINING", True)
ML_THRESHOLD = float(os.getenv("ML_THRESHOLD", 0.55))
ADAPTIVE_ENV_PATH = os.getenv("ADAPTIVE_ENV_PATH", str(BASE_DIR / ".env"))
ML_CLOSE_THRESHOLD = float(os.getenv("ML_CLOSE_THRESHOLD", "0.4"))
STARTUP_GRACE_MINUTES = int(os.getenv("STARTUP_GRACE_MINUTES", "10"))
OPEN_GRACE_MINUTES = int(os.getenv("OPEN_GRACE_MINUTES", "5"))
ML_CLOSE_MIN_CONSECUTIVE = int(os.getenv("ML_CLOSE_MIN_CONSECUTIVE", "2"))
ML_CLOSE_REQUIRE_NEG_PNL = env_bool("ML_CLOSE_REQUIRE_NEG_PNL", False)
ML_THRESHOLD_ALTCOIN = float(os.getenv("ML_THRESHOLD_ALTCOIN", 0.45))
TOP_VOLUME_COUNT = int(os.getenv("TOP_VOLUME_COUNT", 30))

TRAINING_MIN = float(os.getenv("TRAINING_MIN", 0.25))
TRAINING_MAX = float(os.getenv("TRAINING_MAX", 0.45))
TRAINING_POSITION_SIZE = float(os.getenv("TRAINING_POSITION_SIZE", 3.0))
POSITION_SIZING_MODE = os.getenv("POSITION_SIZING_MODE", "risk").lower()
POSITION_PERCENT = float(os.getenv("POSITION_PERCENT", 5.0))
MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", 5.0))
MAX_NOTIONAL_PERCENT = float(os.getenv("MAX_NOTIONAL_PERCENT", 100.0))
TARGET_MARGIN_USDT = float(os.getenv("TARGET_MARGIN_USDT", 0.0))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MIN_SIGNAL_STRENGTH = int(os.getenv("MIN_SIGNAL_STRENGTH", 2))
MAX_SHORT_POSITIONS = int(os.getenv("MAX_SHORT_POSITIONS", 2))
MAX_LONG_POSITIONS = int(os.getenv("MAX_LONG_POSITIONS", 2))
MIN_PRICE = float(os.getenv("MIN_PRICE", 0.50))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 900))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", 120))
PNL_REPORT_INTERVAL = int(os.getenv("PNL_REPORT_INTERVAL", 1800))
SYMBOL_CACHE_TTL = int(os.getenv("SYMBOL_CACHE_TTL_MINUTES", 60))
LIQUIDITY_THRESHOLD = float(os.getenv("LIQUIDITY_THRESHOLD", 10000))
MAX_ACCOUNT_RISK_PERCENT = float(os.getenv("MAX_ACCOUNT_RISK_PERCENT", 2.0))
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML")
working_type = os.getenv("WORKING_PRICE_TYPE", "MARK_PRICE")
use_price_protect = env_bool("PRICE_PROTECT", True)
DEFAULT_REQUIRED_FEATURES = ["rsi","ema_diff","macd_direction","bb_position","volume_ratio","atr_percent"]
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "model_meta.json")
ALLOW_MULTI_ENTRY_PER_SYMBOL = env_bool("ALLOW_MULTI_ENTRY_PER_SYMBOL", False)
REENTRY_COOLDOWN_MIN = int(os.getenv("REENTRY_COOLDOWN_MIN", "15"))
use_limit_entry = env_bool("USE_LIMIT_ENTRY", False)
short_bias = float(os.getenv("SHORT_THRESHOLD_REDUCE", "0.0"))
short_rsi_min = float(os.getenv("SHORT_RSI_MIN", "30"))
short_rsi_max = float(os.getenv("SHORT_RSI_MAX", "75"))
MIN_POSITION_HOLD_MINUTES = int(os.getenv("MIN_POSITION_HOLD_MINUTES", "15"))
COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.0004"))
MIN_PROFIT_THRESHOLD = float(os.getenv("MIN_PROFIT_THRESHOLD", "0.8"))
LIMIT_FILL_TIMEOUT_SEC = int(os.getenv("LIMIT_FILL_TIMEOUT_SEC", "25"))
CANCEL_UNFILLED_LIMIT = env_bool("CANCEL_UNFILLED_LIMIT", True)
MIN_FILLED_RATIO_FOR_ACTIVATION = float(os.getenv("MIN_FILLED_RATIO_FOR_ACTIVATION", "0.5"))
MARK_PENDING_AS_POSITION = env_bool("MARK_PENDING_AS_POSITION", True)
ADAPTIVE_EVAL_INTERVAL = int(os.getenv("ADAPTIVE_EVAL_INTERVAL", "900"))
_last_adapt_eval = None

# --- Global State ---
cooldown_tracker = {}
last_positions_time: Dict[str, datetime] = {}
model_cls = None
model_reg = None
last_scanned = []
STARTUP_AT = datetime.utcnow()
risk_manager = None
market_analyzer = None
adaptive_manager = None

# =========================
# Risk Yönetimi
# =========================
class RiskManager:
    def __init__(self, max_daily_loss_pct=2.0, max_weekly_loss_pct=5.0, starting_capital=None):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.starting_capital = starting_capital
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0)
        self.weekly_reset_time = self._get_week_start()
        self.daily_loss = 0
        self.weekly_loss = 0
        self.load_state()
    def record_trade(self, profit_usdt):
        if profit_usdt < 0:
            self.daily_loss += -profit_usdt
            self.weekly_loss += -profit_usdt
        self.save_state()
    def should_trade(self, account_balance):
        now = datetime.utcnow()
        if now > self.daily_reset_time + timedelta(days=1):
            self.daily_loss = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
        if now > self.weekly_reset_time + timedelta(days=7):
            self.weekly_loss = 0
            self.weekly_reset_time = self._get_week_start()
        balance_to_check = account_balance if self.starting_capital is None else self.starting_capital
        if self.daily_loss > balance_to_check * (self.max_daily_loss_pct / 100):
            return False, "Günlük maksimum kayıp limitine ulaşıldı"
        if self.weekly_loss > balance_to_check * (self.max_weekly_loss_pct / 100):
            return False, "Haftalık maksimum kayıp limitine ulaşıldı"
        return True, None
    def _get_week_start(self):
        now = datetime.utcnow()
        return now - timedelta(days=now.weekday())
    def save_state(self):
        state = {
            "daily_loss": self.daily_loss,
            "weekly_loss": self.weekly_loss,
            "daily_reset": self.daily_reset_time.isoformat(),
            "weekly_reset": self.weekly_reset_time.isoformat()
        }
        atomic_write_json(RISK_STATE_FILE, state)
    def load_state(self):
        try:
            if os.path.exists(RISK_STATE_FILE):
                with open(RISK_STATE_FILE, "r") as f:
                    state = json.load(f)
                self.daily_loss = state.get("daily_loss", 0)
                self.weekly_loss = state.get("weekly_loss", 0)
                self.daily_reset_time = datetime.fromisoformat(
                    state.get("daily_reset", self.daily_reset_time.isoformat()))
                self.weekly_reset_time = datetime.fromisoformat(
                    state.get("weekly_reset", self.weekly_reset_time.isoformat()))
        except Exception:
            pass

# =========================
# Market Analiz
# =========================
class MarketStateAnalyzer:
    def __init__(self):
        self.market_state = "unknown"
        self.volatility_level = "normal"
        self.last_analysis = datetime.utcnow() - timedelta(hours=1)
        self.btc_dominant_coins = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    async def analyze_market(self, client):
        now = datetime.utcnow()
        if (now - self.last_analysis).total_seconds() < 3600:
            return self.market_state, self.volatility_level
        try:
            btc_data = await fetch_klines("BTCUSDT", interval="1h", limit=48)
            if not btc_data:
                return self.market_state, self.volatility_level
            btc_closes = np.array([float(k['close']) for k in btc_data])
            btc_high = np.array([float(k['high']) for k in btc_data])
            btc_low = np.array([float(k['low']) for k in btc_data])
            atr = talib.ATR(btc_high, btc_low, btc_closes, timeperiod=14)[-1]
            change_24h = abs((btc_closes[-1] - btc_closes[-24]) / btc_closes[-24] * 100)
            ema_short = talib.EMA(btc_closes, timeperiod=7)[-1]
            ema_long = talib.EMA(btc_closes, timeperiod=25)[-1]
            if ema_short > ema_long * 1.01:
                self.market_state = "uptrend"
            elif ema_short < ema_long * 0.99:
                self.market_state = "downtrend"
            else:
                self.market_state = "ranging"
            atr_pct = atr / btc_closes[-1] * 100
            if atr_pct > 3 or change_24h > 5:
                self.volatility_level = "high"
            elif atr_pct < 1 and change_24h < 2:
                self.volatility_level = "low"
            else:
                self.volatility_level = "normal"
            self.last_analysis = now
            return self.market_state, self.volatility_level
        except Exception as e:
            logger.error(f"Market analizi hatası: {e}")
            return self.market_state, self.volatility_level
    def get_position_adjustment(self):
        if self.volatility_level == "high":
            return {"size_multiplier": 0.6, "max_positions": 3}
        elif self.volatility_level == "low":
            return {"size_multiplier": 1.2, "max_positions": 6}
        else:
            return {"size_multiplier": 1.0, "max_positions": 5}
    def get_strategy_adjustment(self):
        if self.market_state == "uptrend":
            return {"favor_longs": True, "favor_shorts": False, "rsi_low": 30, "rsi_high": 85}
        elif self.market_state == "downtrend":
            return {"favor_longs": False, "favor_shorts": True, "rsi_low": 25, "rsi_high": 60}
        else:
            return {"favor_longs": True, "favor_shorts": True, "rsi_low": 30, "rsi_high": 70}

# =========================
# Basit Backtester (Değişmedi)
# =========================
class SimpleBacktester:
    def __init__(self, strategy_params=None):
        self.strategy_params = strategy_params or {}
        self.results = []
        self.summary = {}
    async def backtest_symbol(self, symbol, days=30, interval="15m"):
        bars_needed = int((days * 24 * 60) / int(interval[:-1]))
        klines = await fetch_klines(symbol, interval=interval, limit=min(1000, bars_needed))
        if len(klines) < 50:
            return None
        closes = np.array([float(k['close']) for k in klines])
        highs = np.array([float(k['high']) for k in klines])
        lows = np.array([float(k['low']) for k in klines])
        volumes = np.array([float(k['volume']) for k in klines])
        rsi = talib.RSI(closes, 14)
        ema = talib.EMA(closes, 20)
        macd, macd_signal, _ = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        upper_bb, middle_bb, lower_bb = talib.BBANDS(closes, timeperiod=20)
        trades = []
        in_position = False
        entry_price = 0
        entry_index = 0
        side = None
        for i in range(30, len(closes)-1):
            if in_position:
                if side == "long":
                    if highs[i] >= entry_price * 1.015:
                        trades.append({"side": side,"entry_price": entry_price,"exit_price": entry_price*1.015,
                                       "entry_index": entry_index,"exit_index": i,"pnl_pct": 1.5,"exit_type": "tp1"})
                        in_position = False
                    elif lows[i] <= entry_price * 0.99:
                        trades.append({"side": side,"entry_price": entry_price,"exit_price": entry_price*0.99,
                                       "entry_index": entry_index,"exit_index": i,"pnl_pct": -1.0,"exit_type": "sl"})
                        in_position = False
                elif side == "short":
                    if lows[i] <= entry_price * 0.985:
                        trades.append({"side": side,"entry_price": entry_price,"exit_price": entry_price*0.985,
                                       "entry_index": entry_index,"exit_index": i,"pnl_pct": 1.5,"exit_type": "tp1"})
                        in_position = False
                    elif highs[i] >= entry_price * 1.01:
                        trades.append({"side": side,"entry_price": entry_price,"exit_price": entry_price*1.01,
                                       "entry_index": entry_index,"exit_index": i,"pnl_pct": -1.0,"exit_type": "sl"})
                        in_position = False
            else:
                long_cond = (closes[i] > ema[i] and macd[i] > macd_signal[i] and 30 < rsi[i] < 70 and
                             volumes[i] > np.mean(volumes[i-10:i]))
                short_cond = (closes[i] < ema[i] and macd[i] < macd_signal[i] and 30 < rsi[i] < 70 and
                              volumes[i] > np.mean(volumes[i-10:i]))
                if long_cond:
                    in_position = True; side = "long"; entry_price = closes[i]; entry_index = i
                elif short_cond:
                    in_position = True; side = "short"; entry_price = closes[i]; entry_index = i
        if not trades:
            return None
        win_count = sum(1 for t in trades if t["pnl_pct"] > 0)
        loss_count = sum(1 for t in trades if t["pnl_pct"] < 0)
        win_rate = win_count / len(trades)
        return {
            "symbol": symbol,
            "trades": trades,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "wins": win_count,
            "losses": loss_count,
            "avg_profit": np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]) if win_count else 0,
            "avg_loss": np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] < 0]) if loss_count else 0,
            "profit_factor": abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0) /
                                 sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0)) if loss_count else float('inf')
        }
    async def run_multiple_symbols(self, symbols, days=30):
        results = []
        for symbol in symbols:
            r = await self.backtest_symbol(symbol, days)
            if r: results.append(r)
        if results:
            total_trades = sum(r["total_trades"] for r in results)
            total_wins = sum(r["wins"] for r in results)
            self.summary = {
                "symbols_tested": len(results),
                "total_trades": total_trades,
                "win_rate": total_wins / total_trades if total_trades else 0,
                "best_symbols": sorted(results, key=lambda x: x["win_rate"], reverse=True)[:5],
                "worst_symbols": sorted(results, key=lambda x: x["win_rate"])[:5],
                "avg_trades_per_symbol": total_trades / len(results) if results else 0
            }
        self.results = results
        return self.summary

# =========================
# Yardımcı IO Fonksiyonları
# =========================
def atomic_write_json(path: str, data: object):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)

def initialize_files():
    for file in [POSITION_FILE, HISTORY_FILE, SYMBOL_CACHE_FILE, BLACKLIST_FILE, RISK_STATE_FILE]:
        if not os.path.exists(file):
            if file == SYMBOL_CACHE_FILE:
                atomic_write_json(file, {"timestamp": datetime.utcnow().isoformat(), "symbols": []})
            elif file == RISK_STATE_FILE:
                atomic_write_json(file, {
                    "daily_loss": 0,
                    "weekly_loss": 0,
                    "daily_reset": datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat(),
                    "weekly_reset": (datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())).isoformat()
                })
            else:
                atomic_write_json(file, [])

def load_config(config_path=CONFIG_FILE):
    default_config = {
        "risk": {"max_daily_loss_percent": 2.0, "max_weekly_loss_percent": 5.0},
        "general": {"max_positions": 5, "cooldown_minutes": 120},
        "filters": {"min_volume_usdt": 10000},
        "ml": {"threshold": 0.55}
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            for cat in default_config:
                if cat in loaded and isinstance(loaded[cat], dict):
                    default_config[cat].update(loaded[cat])
            return default_config
        except Exception as e:
            logger.error(f"Config yükleme hatası: {e}")
    return default_config

def add_to_blacklist(symbol):
    try:
        blacklist = load_blacklist()
        if symbol not in blacklist:
            blacklist.append(symbol)
            atomic_write_json(BLACKLIST_FILE, blacklist)
            logger.info(f"{symbol} blacklist'e eklendi.")
    except Exception as e:
        logger.error(f"{symbol} blacklist'e eklenemedi: {e}")

def load_blacklist():
    try:
        with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def load_symbol_cache():
    if not os.path.exists(SYMBOL_CACHE_FILE):
        return None
    try:
        with open(SYMBOL_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if "timestamp" not in cache or "symbols" not in cache:
            logger.warning("Symbol cache yapısı eksik. Yeniden oluşturulacak.")
            return None
        timestamp = datetime.fromisoformat(cache["timestamp"])
        if (datetime.utcnow() - timestamp).total_seconds() > SYMBOL_CACHE_TTL * 60:
            return None
        return cache["symbols"]
    except Exception as e:
        logger.error(f"Cache yükleme hatası: {e}")
        return None

def save_symbol_cache(symbols):
    try:
        cache = {"timestamp": datetime.utcnow().isoformat(), "symbols": symbols}
        atomic_write_json(SYMBOL_CACHE_FILE, cache)
    except Exception as e:
        logger.error(f"Symbol cache kaydetme hatası: {e}")

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def load_model_meta_safe():
    try:
        if Path(MODEL_META_PATH).exists():
            return json.loads(Path(MODEL_META_PATH).read_text(encoding="utf-8") or "{}")
    except Exception:
        pass
    return {}

MODEL_META = load_model_meta_safe()

def get_required_features():
    env_feats = os.getenv("ML_FEATURES", "").strip()
    if env_feats:
        names = [x.strip() for x in env_feats.split(",") if x.strip()]
        if names:
            return names
    meta_names = MODEL_META.get("feature_names")
    if isinstance(meta_names, list) and meta_names:
        return meta_names
    return DEFAULT_REQUIRED_FEATURES

REQUIRED_FEATURES = get_required_features()

def get_ml_threshold(default_val: float = 0.45) -> float:
    try:
        if "ML_THRESHOLD" in os.environ:
            return float(os.getenv("ML_THRESHOLD"))
    except Exception:
        pass
    try:
        rec = MODEL_META.get("recommended_threshold")
        if rec is not None:
            return float(rec)
    except Exception:
        pass
    return default_val

def get_dynamic_ml_threshold(symbol):
    base_threshold = float(os.getenv("ML_THRESHOLD", 0.5))
    altcoin_threshold = float(os.getenv("ML_THRESHOLD_ALTCOIN", 0.45))
    global market_analyzer
    if market_analyzer:
        volatility = market_analyzer.volatility_level
        if volatility == "high":
            base_threshold += 0.03
            altcoin_threshold += 0.03
        elif volatility == "low":
            base_threshold -= 0.02
            altcoin_threshold -= 0.02
    if symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]:
        return base_threshold
    elif symbol in ["DOGEUSDT", "SHIBUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT"]:
        return base_threshold - 0.03
    else:
        return altcoin_threshold

def is_invert_prob() -> bool:
    env_val = os.getenv("INVERT_ML_PROB", "")
    if env_val != "":
        return str(env_val).strip().lower() in ("1","true","yes","y","on")
    return bool(MODEL_META.get("inverted", False))

def select_features_frame(df_or_dict):
    if isinstance(df_or_dict, dict):
        row = {k: df_or_dict.get(k, np.nan) for k in REQUIRED_FEATURES}
        df = pd.DataFrame([row], columns=REQUIRED_FEATURES)
    else:
        df = df_or_dict.reindex(columns=REQUIRED_FEATURES)
    if "bb_position" in df.columns:
        df["bb_position_squared"] = df["bb_position"] ** 2
    if "rsi" in df.columns:
        df["rsi_normalized"] = (df["rsi"] - 50) / 50
    if all(c in df.columns for c in ["bb_position", "rsi", "rsi_normalized"]):
        df["bb_rsi_ratio"] = df["bb_position"] * df["rsi_normalized"]
    if "volume_ratio" in df.columns and "atr_percent" in df.columns:
        df["volume_volatility_ratio"] = df["volume_ratio"] / (df["atr_percent"] + 0.1)
    if "ema_diff" in df.columns and "macd_direction" in df.columns:
        df["trend_strength"] = df["ema_diff"] * df["macd_direction"]
    return df.fillna(0)

def getenv_any(keys, default=None):
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return v
    return default

def tg_html(s) -> str:
    return _escape(str(s), quote=False)

def _strip_basic_md(text: str) -> str:
    return re.sub(r'([_*`])', '', str(text))

async def send_telegram_message(text: str):
    token = getenv_any(["TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "BOT_TOKEN"])
    chat_id = getenv_any(["TELEGRAM_CHAT_ID", "CHAT_ID"])
    if not token or not chat_id:
        logger.debug("Telegram devre dışı (token/chat_id yok).")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    parse_mode = None if TELEGRAM_PARSE_MODE.lower() == "plain" else TELEGRAM_PARSE_MODE
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        async with aiohttp.ClientSession() as session:
            resp = await session.post(url, json=payload)
            body = await resp.text()
            if resp.status == 200:
                return
            if "can't parse entities" in body.lower() or "parse" in body.lower():
                fallback = {"chat_id": chat_id,"text": _strip_basic_md(text),"disable_web_page_preview": True}
                resp2 = await session.post(url, json=fallback)
                if resp2.status != 200:
                    logger.error(f"Telegram fallback error: {await resp2.text()}")
            else:
                logger.error(f"Telegram error: {body}")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

def compute_unrealized_pnl_pct(side: str, entry: float, mark: float) -> float:
    if not entry or not mark:
        return 0.0
    if (side or "long").lower() == "long":
        return (mark - entry) / entry * 100.0
    return (entry - mark) / entry * 100.0

async def should_ml_close(pos: dict, ml_prob: float, consec_low_cnt: int, atr_pct: float, mark_px: float) -> tuple[bool, str]:
    ml_th = env_float("ML_THRESHOLD", 0.45)
    grace_min = env_int("ML_CLOSE_GRACE_MIN", 15)
    need_consec = env_int("ML_CLOSE_CONSEC", 3)
    min_adverse = env_float("ML_CLOSE_MIN_ADVERSE_PCT", 0.4)
    atr_guard = env_float("ML_CLOSE_ATR_GUARD", 0.5)
    skip_if_be = env_bool("ML_CLOSE_SKIP_IF_BE_PROTECTED", True)
    if ml_prob >= ml_th or consec_low_cnt < need_consec:
        return False, "ml_ok_or_not_enough_consec"
    opened_at = pos.get("opened_at") or pos.get("timestamp")
    if opened_at:
        try:
            t0 = datetime.strptime(opened_at, "%Y-%m-%d %H:%M:%S")
            if (datetime.utcnow() - t0).total_seconds() < grace_min * 60:
                return False, "grace_period"
        except Exception:
            pass
    if skip_if_be and pos.get("tp1_hit"):
        entry_px = float(pos.get("entry_price") or 0.0)
        side = pos.get("side", "long")
        be_offset_bp = env_float("TP1_BE_OFFSET_BP", 3.0)
        be_px = be_price_from_entry(entry_px, side, be_offset_bp)
        if entry_px > 0 and be_px > 0 and mark_px > 0:
            if (side == "long" and mark_px >= be_px) or (side == "short" and mark_px <= be_px):
                return False, "protected_by_BE"
    entry_px = float(pos.get("entry_price") or 0.0)
    side = pos.get("side", "long")
    pnl_pct = compute_unrealized_pnl_pct(side, entry_px, mark_px)
    if pnl_pct > -min_adverse:
        return False, f"pnl_not_bad({pnl_pct:.2f}%)"
    if atr_pct and atr_pct > 0:
        adverse_move_pct = -pnl_pct
        threshold_pct = atr_pct * atr_guard
        if adverse_move_pct < threshold_pct:
            return False, f"inside_atr({adverse_move_pct:.2f}%<{threshold_pct:.2f}%)"
    return True, "ml_low_prob_with_adverse_move"

def be_price_from_entry(entry: float, side: str, offset_bp: float) -> float:
    if entry <= 0:
        return entry
    m = offset_bp / 10000.0
    if (side or "long").lower() == "long":
        return entry * (1.0 + m)
    return entry * (1.0 - m)

def ema(values: list, period: int) -> Optional[float]:
    if not values or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

# =========================
# ADAPTİF ML THRESHOLD
# =========================
def _parse_float_list_env(key: str, default: List[float]) -> List[float]:
    raw = os.getenv(key, "")
    if not raw:
        return default
    out = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except:
            pass
    return sorted(set(out)) if out else default

def _safe_expected_value(win_rate: float, avg_win: float, avg_loss: float) -> float:
    return win_rate * avg_win + (1 - win_rate) * avg_loss

def _calculate_metrics(trades: List[dict]) -> dict:
    if not trades:
        return {"count":0,"win_rate":0,"profit_factor":0,"avg_profit":0,"avg_win":0,
                "avg_loss":0,"sum_profit":0,"expected_value":0}
    profits = [t["profit"] for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    win_rate = len(wins)/len(profits) if profits else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    loss_abs_sum = abs(sum(losses)) if losses else 0
    win_sum = sum(wins)
    pf = (win_sum / loss_abs_sum) if loss_abs_sum > 0 else (float("inf") if wins else 0)
    expected_value = _safe_expected_value(win_rate, avg_win, avg_loss)
    return {
        "count": len(profits),
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_profit": np.mean(profits),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sum_profit": sum(profits),
        "expected_value": expected_value
    }

def _load_recent_trades(window: int) -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist = json.load(f)
    except Exception as e:
        logger.error(f"[ADAPT-ML] History okunamadı: {e}")
        return []
    valid = []
    for r in hist:
        if r.get("closed_reason") == "shadow":
            continue
        if r.get("profit_usdt") is None or r.get("ml_probability") is None:
            continue
        try:
            profit = float(r.get("profit_usdt"))
            prob = float(r.get("ml_probability"))
        except:
            continue
        valid.append({"timestamp": r.get("timestamp"), "profit": profit, "prob": prob})
    def _dt(x):
        try:
            return datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.utcnow()
    valid.sort(key=_dt)
    return valid[-window:] if window > 0 else valid

def _pick_best_threshold(trades: List[dict],
                         thresholds: List[float],
                         min_support: int,
                         winrate_min: float,
                         pf_min: float,
                         ev_min: float) -> Optional[dict]:
    if not trades:
        return None
    candidates = []
    for thr in thresholds:
        subset = [t for t in trades if t["prob"] >= thr]
        m = _calculate_metrics(subset)
        m["support"] = m["count"]
        if m["support"] == 0:
            continue
        passed = (m["support"] >= min_support and
                  m["win_rate"] >= winrate_min and
                  m["profit_factor"] >= pf_min and
                  m["expected_value"] >= ev_min)
        m["passed_filters"] = passed
        candidates.append({"threshold": thr, "metrics": m})
    if not candidates:
        return None
    filtered = [c for c in candidates if c["metrics"]["passed_filters"]]
    pool = filtered if filtered else candidates
    pool.sort(key=lambda c: (c["metrics"]["expected_value"],
                             c["metrics"]["profit_factor"],
                             c["metrics"]["sum_profit"]), reverse=True)
    return pool[0] if pool else None

def _update_env_file(env_path: str, key: str, value: str) -> bool:
    try:
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        key_line_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.split("=", 1)[0] == key:
                key_line_idx = i
                break
        new_line = f"{key}={value}"
        if key_line_idx is not None:
            lines[key_line_idx] = new_line
        else:
            lines.append(new_line)
        tmp_path = env_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp_path, env_path)
        return True
    except Exception as e:
        logger.error(f"[ADAPT-ML] .env güncelleme hatası: {e}")
        return False

# Adaptif threshold state
_last_adaptive_run = None
_current_adaptive_threshold = None
_last_adaptive_good_update = None
_last_adaptive_raw_pf = None
_last_adaptive_raw_ev = None

async def run_adaptive_ml_threshold(
    force: bool = False,
    send_telegram: bool = True
) -> Optional[float]:
    """
    Guard'lı adaptif ML threshold hesaplama fonksiyonu.
    """
    global _last_adaptive_run, _current_adaptive_threshold
    global _last_adaptive_good_update, _last_adaptive_raw_pf, _last_adaptive_raw_ev

    if not env_bool("ADAPTIVE_ML_THRESHOLD_ENABLE", False):
        return None
    now = datetime.utcnow()
    refresh_hours = float(os.getenv("ADAPTIVE_ML_THRESHOLD_REFRESH_HOURS", "6"))
    min_hours_forced_fallback = float(os.getenv("ADAPTIVE_ML_THRESHOLD_FORCE_MIN_HOURS", "4"))
    allow_fallback = env_bool("ADAPTIVE_ML_THRESHOLD_ALLOW_FALLBACK", False)
    if (not force) and _last_adaptive_run and (now - _last_adaptive_run).total_seconds() < refresh_hours * 3600:
        return None
    window = int(os.getenv("ADAPTIVE_ML_THRESHOLD_WINDOW", "150"))
    min_trades_total = int(os.getenv("ADAPTIVE_ML_THRESHOLD_MIN_TRADES", "120"))
    thresholds = _parse_float_list_env(
        "ADAPTIVE_ML_THRESHOLD_CANDIDATES",
        [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72]
    )
    min_support = int(os.getenv("ADAPTIVE_ML_THRESHOLD_MIN_SUPPORT", "25"))
    winrate_min = float(os.getenv("ADAPTIVE_ML_THRESHOLD_WINRATE_MIN", "0.52"))
    pf_min = float(os.getenv("ADAPTIVE_ML_THRESHOLD_PROFIT_FACTOR_MIN", "1.05"))
    ev_min = float(os.getenv("ADAPTIVE_ML_THRESHOLD_EV_MIN", "0"))
    min_change = float(os.getenv("ADAPTIVE_ML_THRESHOLD_MIN_CHANGE", "0.01"))
    smoothing = float(os.getenv("ADAPTIVE_ML_THRESHOLD_SMOOTHING", "0.5"))
    thr_min = float(os.getenv("ADAPTIVE_ML_THRESHOLD_MIN", "0.45"))
    thr_max = float(os.getenv("ADAPTIVE_ML_THRESHOLD_MAX", "0.80"))
    trades = _load_recent_trades(window)
    if len(trades) < min_trades_total:
        if send_telegram:
            await send_telegram_message(
                f"ℹ️ <b>Adaptive ML</b>: Veri yetersiz (son={len(trades)}/{min_trades_total}) – güncelleme yok."
            )
        _last_adaptive_run = now
        return None
    best = _pick_best_threshold(trades, thresholds, min_support, winrate_min, pf_min, ev_min)
    if not best:
        if send_telegram:
            await send_telegram_message(
                "ℹ️ <b>Adaptive ML</b>: Geçerli aday yok (filtreler sıkı) – güncelleme yok."
            )
        _last_adaptive_run = now
        return None
    raw_best = best["threshold"]
    metrics = best["metrics"]
    support = metrics.get("support", 0)
    pf_raw = metrics.get("profit_factor", 0)
    ev_raw = metrics.get("expected_value", 0)
    passed_filters = (
        support >= min_support and
        metrics["win_rate"] >= winrate_min and
        metrics["profit_factor"] >= pf_min and
        metrics["expected_value"] >= ev_min
    )
    is_fallback = not passed_filters
    if support < min_support:
        if send_telegram:
            await send_telegram_message(
                f"⏸ <b>Adaptive ML</b>: Destek yetersiz (support={support}<{min_support})."
            )
        _last_adaptive_run = now
        return None
    if pf_raw < 1.0 or ev_raw <= 0:
        if send_telegram:
            await send_telegram_message(
                f"⏸ <b>Adaptive ML</b>: Kalite düşük (PF={pf_raw:.2f}, EV={ev_raw:.5f})."
            )
        _last_adaptive_run = now
        return None
    if is_fallback and not allow_fallback:
        if _last_adaptive_good_update:
            hours_since_good = (now - _last_adaptive_good_update).total_seconds()/3600
        else:
            hours_since_good = 999
        if hours_since_good < min_hours_forced_fallback:
            if send_telegram:
                await send_telegram_message(
                    f"⏸ <b>Adaptive ML</b>: Fallback engellendi (PF={pf_raw:.2f}) – son iyi güncelleme {hours_since_good:.1f}h"
                )
            _last_adaptive_run = now
            return None
    try:
        current_thr_env = float(os.getenv("ML_THRESHOLD", "0.6"))
    except:
        current_thr_env = 0.6
    base_prev = _current_adaptive_threshold if _current_adaptive_threshold is not None else current_thr_env
    if abs(raw_best - base_prev) < 0.005:
        if send_telegram:
            await send_telegram_message(
                f"ℹ️ <b>Adaptive ML</b>: Raw yakın (Δ={abs(raw_best-base_prev):.4f}) – güncellenmedi."
            )
        _last_adaptive_run = now
        return None
    new_thr_smoothed = base_prev * smoothing + raw_best * (1 - smoothing)
    new_thr_smoothed = max(thr_min, min(thr_max, new_thr_smoothed))
    if abs(new_thr_smoothed - current_thr_env) < min_change:
        if send_telegram:
            await send_telegram_message(
                f"ℹ️ <b>Adaptive ML</b>: Değişim küçük (Δ={abs(new_thr_smoothed-current_thr_env):.4f})."
            )
        _current_adaptive_threshold = new_thr_smoothed
        _last_adaptive_run = now
        return None
    env_updated = _update_env_file(ADAPTIVE_ENV_PATH, "ML_THRESHOLD", f"{new_thr_smoothed:.4f}")
    os.environ["ML_THRESHOLD"] = f"{new_thr_smoothed:.4f}"
    _current_adaptive_threshold = new_thr_smoothed
    _last_adaptive_run = now
    if pf_raw >= pf_min and ev_raw > 0:
        _last_adaptive_good_update = now
    _last_adaptive_raw_pf = pf_raw
    _last_adaptive_raw_ev = ev_raw
    status_prefix = "🤖 Adaptive ML Threshold Güncellendi"
    if is_fallback:
        status_prefix = "⚠️ Adaptive ML (Fallback) Güncelleme"
    msg_lines = [
        status_prefix,
        f"• Raw: <code>{raw_best:.3f}</code>",
        f"• Smooth: <code>{new_thr_smoothed:.3f}</code>",
        f"• Önceki: <code>{current_thr_env:.3f}</code>",
        f"• Support: <code>{support}</code>",
        f"• WinRate: <code>{metrics['win_rate']*100:.2f}%</code>",
        f"• PF: <code>{pf_raw:.2f}</code>",
        f"• EV: <code>{ev_raw:.5f}</code>",
        f"• SumProfit: <code>{metrics['sum_profit']:.4f}</code>",
        f"• Passed: <code>{'Evet' if passed_filters else 'Hayır'}</code>",
        f"• Fallback: <code>{'Evet' if is_fallback else 'Hayır'}</code>",
        f"• ENV Güncellendi: <code>{'Evet' if env_updated else 'Hayır'}</code>"
    ]
    if send_telegram:
        await send_telegram_message("\n".join(msg_lines))
    logger.info("[ADAPT-ML] Threshold -> %.4f (raw=%.4f, prev=%.4f, pf=%.2f, ev=%.5f, fallback=%s)",
                new_thr_smoothed, raw_best, current_thr_env, pf_raw, ev_raw, is_fallback)
    return new_thr_smoothed

async def adaptive_ml_threshold_loop():
    if not env_bool("ADAPTIVE_ML_THRESHOLD_ENABLE", False):
        logger.info("[ADAPT-ML] Kapalı.")
        return
    try:
        await run_adaptive_ml_threshold(force=True, send_telegram=True)
    except Exception as e:
        logger.error(f"[ADAPT-ML] İlk çalıştırma hatası: {e}")
    refresh_hours = float(os.getenv("ADAPTIVE_ML_THRESHOLD_REFRESH_HOURS", "6"))
    interval = max(1800, int(refresh_hours * 3600 / 2))
    while True:
        try:
            await run_adaptive_ml_threshold(force=False, send_telegram=True)
        except Exception as e:
            logger.error(f"[ADAPT-ML] Döngü hatası: {e}")
        await asyncio.sleep(interval)

async def entry_soft_filters(symbol: str, side: str, entry_price: float) -> Tuple[bool, str]:
    tf = os.getenv("ENTRY_TF", "5m")
    ext_max = env_float("EXT_MAX_DEV_PCT", 1.5)
    lookback = int(os.getenv("PULLBACK_LOOKBACK", "10"))
    need_touches = int(os.getenv("MIN_PULLBACK_BARS", "2"))
    spike_lb = int(os.getenv("SPIKE_LOOKBACK", "12"))
    spike_max = env_float("SPIKE_MAX_PCT", 5.0)
    use_spike_guard = env_bool("USE_SPIKE_GUARD", True)

    kl = await fetch_klines(symbol, interval=tf, limit=max(lookback + 50, 80))
    if not kl or len(kl) < max(lookback + 1, 30):
        return True, "no_data_softpass"

    closes = [float(k['close']) for k in kl]
    highs = [float(k['high']) for k in kl]
    lows  = [float(k['low']) for k in kl]

    ema_period = 20
    ema_val = ema(closes[-(ema_period+5):], ema_period)
    if not ema_val or ema_val <= 0:
        return True, "ema_na_softpass"
    dev_pct = abs(entry_price - ema_val) / ema_val * 100.0
    if dev_pct > ext_max:
        return False, f"extension>{ext_max}%"

    last_lows = lows[-lookback:]
    last_highs = highs[-lookback:]
    touches = 0
    if (side or "long").lower() == "long":
        touches = sum(1 for x in last_lows if x <= ema_val)
    else:
        touches = sum(1 for x in last_highs if x >= ema_val)
    if touches < need_touches:
        return False, f"pullback<{need_touches}"

    if use_spike_guard and spike_lb >= 2:
        ref_close_old = closes[-spike_lb]
        ref_close_new = closes[-1]
        move_pct = abs(ref_close_new - ref_close_old) / ref_close_old * 100.0 if ref_close_old > 0 else 0.0
        if move_pct >= spike_max and dev_pct > (ext_max * 0.6):
            return False, f"spike>{spike_max}%"

    return True, "ok"

async def check_pullback_entry(symbol, side, entry_price):
    """Fiyatın pullback yaşayıp yaşamadığını kontrol eder"""
    if not env_bool("USE_PULLBACK_ENTRY", False):
        return True, "pullback_check_disabled"
    
    try:
        tf = os.getenv("ENTRY_TF", "5m")
        bars = await fetch_klines(symbol, interval=tf, limit=20)
        if not bars or len(bars) < 10:
            return True, "insufficient_data"
            
        highs = np.array([float(k['high']) for k in bars])
        lows = np.array([float(k['low']) for k in bars])
        
        if side == "long":
            # Son 3-5 mumda düşüş (pullback) oldu mu?
            recent_high = np.max(highs[-10:-3])
            recent_low = np.min(lows[-3:])
            pullback_pct = (recent_high - recent_low) / recent_high * 100
            
            if pullback_pct < 0.4:  # %0.5'ten az pullback
                return False, f"insufficient_pullback_{pullback_pct:.2f}%"
                
            # Son mumda dönüş işareti var mı?
            if bars[-1]['close'] < bars[-1]['open']:
                return False, "bearish_current_candle"
                
            return True, "valid_pullback"
            
        else:  # short
            # Son 3-5 mumda yükseliş (pullback) oldu mu?
            recent_low = np.min(lows[-10:-3])
            recent_high = np.max(highs[-3:])
            pullback_pct = (recent_high - recent_low) / recent_low * 100
            
            if pullback_pct < 0.5:  # %0.5'ten az pullback
                return False, f"insufficient_pullback_{pullback_pct:.2f}%"
                
            # Son mumda dönüş işareti var mı?
            if bars[-1]['close'] > bars[-1]['open']:
                return False, "bullish_current_candle"
                
            return True, "valid_pullback"
            
    except Exception as e:
        logger.error(f"Pullback kontrolü hatası ({symbol}): {e}")
        return True, "pullback_check_error"

async def should_close_position_early(pos: dict, current_price: float) -> tuple[bool, str]:
    """Pozisyonun erken kapanıp kapanmayacağını kontrol eder"""
    
    # Minimum tutma süresi kontrolü
    opened_at = pos.get("timestamp")
    if opened_at:
        try:
            open_time = datetime.strptime(opened_at, "%Y-%m-%d %H:%M:%S")
            hold_minutes = (datetime.utcnow() - open_time).total_seconds() / 60
            
            if hold_minutes < MIN_POSITION_HOLD_MINUTES:
                return False, f"min_hold_time({hold_minutes:.1f}m<{MIN_POSITION_HOLD_MINUTES}m)"
        except:
            pass
    
    # Komisyon dahil kar kontrolü
    entry_price = float(pos.get("entry_price", 0))
    side = pos.get("side", "long")
    size_usdt = float(pos.get("size", 0))
    
    if entry_price <= 0 or size_usdt <= 0:
        return True, "invalid_position_data"
    
    # PnL hesapla
    if side == "long":
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        pnl_pct = ((entry_price - current_price) / entry_price) * 100
    
    # Komisyon maliyeti (giriş + çıkış)
    commission_cost_pct = (COMMISSION_RATE * 2) * 100  # %0.08
    
    # Net kar
    net_profit_pct = pnl_pct - commission_cost_pct
    
    # Minimum kar eşiğini aştı mı?
    if net_profit_pct < MIN_PROFIT_THRESHOLD:
        return False, f"below_profit_threshold({net_profit_pct:.3f}%<{MIN_PROFIT_THRESHOLD}%)"
    
    return True, "profit_threshold_met"

def _order_num(o: dict, keys: List[str], default: float = None) -> Optional[float]:
    for k in keys:
        if k in o and o[k] is not None:
            try:
                return float(o[k])
            except Exception:
                continue
    return default

def _almost_equal(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol

def _trailing_matches(trailing_orders: List[dict], qty_expected: float, activation_expected: Optional[float], callback_expected: float, tick_size: float, step_size: float) -> bool:
    if not trailing_orders:
        return False

    order = sorted(trailing_orders, key=lambda o: int(o.get("updateTime", 0)), reverse=True)[0]

    qty_order = _order_num(order, ["origQty", "quantity"], default=None)
    if qty_order is None:
        return False
    if not _almost_equal(qty_order, qty_expected, tol=max(step_size, 1e-12)):
        return False

    act_order = _order_num(order, ["activationPrice", "activatePrice", "activatePrice"], default=None)
    if activation_expected is not None:
        if not _almost_equal(act_order, activation_expected, tol=max(tick_size, 1e-12)):
            return False

    cb_order = _order_num(order, ["callbackRate", "priceRate"], default=None)
    if cb_order is None or not _almost_equal(cb_order, round(callback_expected, 2), tol=0.01 + 1e-9):
        return False

    return True

async def prune_local_positions_not_on_exchange(send_notice: bool = True) -> None:
    client = await init_binance_client()
    if not client:
        return
    try:
        account = await client.futures_account()
        ex_open = {
            p.get("symbol") for p in account.get("positions", [])
            if p.get("symbol") and abs(float(p.get("positionAmt") or 0.0)) > 0.0
        }

        positions = load_positions()
        if not positions:
            return

        updated = []
        cleaned = []
        for pos in positions:
            if pos.get("closed", False):
                continue
            sym = pos.get("symbol")
            if sym in ex_open:
                updated.append(pos)
                continue

            try:
                data = await fetch_klines(sym, limit=1)
                last_price = data[-1]['close'] if data else float(pos.get("entry_price", 0.0) or 0.0)
                record_closed_trade(pos, last_price, "Sync cleanup (borsada pozisyon yok)")
                cleaned.append(sym)
            except Exception as e:
                logger.error(f"{sym}: prune cleanup hata: {e}")

        save_positions(updated)

        if send_notice and cleaned:
            txt = ["🧹 <b>Sync Temizliği</b>: Borsada bulunmayan local pozisyonlar kapatıldı."]
            for s in cleaned:
                txt.append(f"• <code>{tg_html(s)}</code>")
            await send_telegram_message("\n".join(txt))

    except Exception as e:
        logger.error(f"prune_local_positions_not_on_exchange hata: {e}", exc_info=True)
    finally:
        try:
            await client.close_connection()
        except:
            pass

# =========================
# Binance Client
# =========================
async def init_binance_client():
    retries = int(os.getenv("MAX_BINANCE_RETRIES", "3"))
    delay = float(os.getenv("BINANCE_RETRY_DELAY", "1.0"))
    last_err = None
    for i in range(retries):
        try:
            testnet_str = os.getenv('BINANCE_TESTNET', 'False').lower()
            testnet = testnet_str in ['1', 'true', 'yes']
            return await asyncio.wait_for(AsyncClient.create(
                api_key=os.getenv('BINANCE_FUTURES_API_KEY'),
                api_secret=os.getenv('BINANCE_FUTURES_SECRET_KEY'),
                testnet=testnet
            ), timeout=API_TIMEOUT)
        except asyncio.TimeoutError as e:
            last_err = e
            logger.error(f"init_binance_client: Binance client açılırken timeout oluştu (deneme {i+1}/{retries})")
        except Exception as e:
            last_err = e
            logger.error(f"Binance client hatası (deneme {i+1}/{retries}): {e}")
        await asyncio.sleep(delay * (i + 1))
    logger.error(f"init_binance_client: tüm denemeler başarısız: {last_err}")
    return None

async def open_binance_limit_position(symbol, side, quantity, limit_price):
    client = await init_binance_client()
    try:
        if not client:
            return None
        order = await client.futures_create_order(
            symbol=symbol,
            side="BUY" if side == "long" else "SELL",
            type="LIMIT",
            quantity=quantity,
            price=limit_price,
            timeInForce="GTC"
        )
        logger.info(f"Limit emir açıldı: {order}")
        return order
    except Exception as e:
        logger.error(f"Binance limit order açma hatası: {e}")
        return None
    finally:
        if client:
            await client.close_connection()

# =========================
# ML Model
# =========================
def train_cls_from_history(min_records: int = 200) -> Optional[RandomForestClassifier]:
    try:
        if not os.path.exists(HISTORY_FILE):
            logger.warning("History dosyası yok, model eğitilemedi.")
            return None
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        if len(history) < min_records:
            logger.warning(f"History yetersiz ({len(history)}/{min_records}), model eğitilemedi.")
            return None
        df = pd.DataFrame(history)
        df = df[df['profit_usdt'].notna()]
        if df.empty:
            logger.warning("History'de profit_usdt yok veya boş, model eğitilemedi.")
            return None
        X = df[REQUIRED_FEATURES]
        y = (df['profit_usdt'] > 0).astype(int)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X, y)
        joblib.dump(model, MODEL_CLASSIFICATION_PATH)
        logger.info(f"✅ Sınıflandırma modeli history'den eğitildi ve kaydedildi: {MODEL_CLASSIFICATION_PATH}")
        return model
    except Exception as e:
        logger.error(f"Model eğitimi hatası: {e}")
        return None

def load_models():
    global model_cls, model_reg
    logger.info(f"Model yol(ları): CLS={MODEL_CLASSIFICATION_PATH}, REG={MODEL_REGRESSION_PATH}")
    logger.info(f"ML Ayarları: threshold={get_ml_threshold(ML_THRESHOLD):.3f}, invert={is_invert_prob()}, features={REQUIRED_FEATURES}")
    try:
        model_cls = joblib.load(MODEL_CLASSIFICATION_PATH)
        logger.info("✅ Sınıflandırma modeli yüklendi")
    except Exception as e:
        logger.warning(f"⚠️ Sınıflandırma modeli yüklenemedi ({MODEL_CLASSIFICATION_PATH}): {e}")
        model_cls = None
        if DISABLE_STARTUP_TRAINING:
            logger.info("Açılışta eğitim devre dışı (DISABLE_STARTUP_TRAINING=1). ML olmadan devam edilecek.")
    try:
        model_reg = joblib.load(MODEL_REGRESSION_PATH)
        logger.info("✅ Regresyon modeli yüklendi")
    except Exception as e:
        logger.warning(f"⚠️ Regresyon modeli yüklenemedi: {e}")
        model_reg = None

class ModelMonitor:
    def __init__(self):
        self.performance_log = []

    def check_decay(self) -> bool:
        if not AUTO_RETRAIN:
            return False
        valid = [x['accuracy'] for x in self.performance_log if x['accuracy'] is not None]
        if len(valid) < 10:
            return False
        last_5 = np.mean(valid[-5:])
        first_5 = np.mean(valid[:5])
        return (first_5 - last_5) > float(os.getenv("MODEL_DECAY_THRESHOLD", 0.15))

    async def retrain_model(self):
        try:
            if not os.path.exists(HISTORY_FILE):
                logger.error("History dosyası bulunamadı!")
                return False
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            if len(history) < 100:
                logger.error(f"Yetersiz veri: {len(history)} kayıt")
                return False
            df = pd.DataFrame(history)
            df = df[df['profit_usdt'].notna()]
            X = df[REQUIRED_FEATURES]
            y = (df['profit_usdt'] > 0).astype(int)
            new_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
            new_model.fit(X, y)
            train_acc = new_model.score(X, y)
            logger.info(f"Retrain başarılı! Doğruluk: {train_acc:.2%}")
            global model_cls
            model_cls = new_model
            joblib.dump(new_model, MODEL_CLASSIFICATION_PATH)
            await send_telegram_message(
                f"🔄 Model yeniden eğitildi!\n"
                f"• Doğruluk: {train_acc:.2%}\n"
                f"• Kullanılan Veri: {len(df)} işlem"
            )
            return True
        except Exception as e:
            error_msg = f"Retrain hatası: {str(e)}"
            logger.error(error_msg)
            await send_telegram_message(f"🔴 {error_msg}")
            return False

model_monitor = ModelMonitor()

# =========================
# Binance Filtreleri ve Yardımcılar
# =========================
async def get_symbol_trading_filters(symbol):
    client = await init_binance_client()
    try:
        if not client:
            return 3, 0.001, 0.0, float(os.getenv("MIN_NOTIONAL_USDT", 5.0)), 0.0001
        info = await client.futures_exchange_info()
        precision, step_size, min_qty = 3, 0.001, 0.0
        min_notional = float(os.getenv("MIN_NOTIONAL_USDT", 5.0))
        tick_size = 0.0001

        for s in info.get('symbols', []):
            if s.get('symbol') == symbol:
                for f in s.get('filters', []):
                    ftype = f.get('filterType')
                    if ftype == 'LOT_SIZE':
                        step_size = float(f.get('stepSize', 0.001))
                        min_qty = float(f.get('minQty', 0.0))
                        try:
                            precision = abs(Decimal(str(step_size)).as_tuple().exponent)
                        except Exception:
                            precision = 3
                    elif ftype == 'MIN_NOTIONAL':
                        mn = f.get('notional') or f.get('minNotional')
                        if mn is not None:
                            min_notional = float(mn)
                    elif ftype == 'PRICE_FILTER':
                        ts = f.get('tickSize')
                        if ts is not None:
                            tick_size = float(ts)
                return precision, step_size, min_qty, min_notional, tick_size

        return precision, step_size, min_qty, min_notional, tick_size
    except Exception as e:
        logger.error(f"Trading filtreleri çekme hatası: {e}")
        return 3, 0.001, 0.0, float(os.getenv("MIN_NOTIONAL_USDT", 5.0)), 0.0001
    finally:
        if client:
            await client.close_connection()

def adjust_quantity_up(quantity, precision, step_size):
    quant = Decimal(str(quantity))
    step = Decimal(str(step_size))
    steps_up = (quant / step).to_integral_value(rounding=ROUND_UP)
    quant = steps_up * step
    quant = quant.quantize(Decimal('1.' + '0' * int(abs(Decimal(str(step_size)).as_tuple().exponent))), rounding=ROUND_DOWN)
    return float(quant)

def adjust_price_to_tick(price: float, tick_size: float) -> float:
    try:
        if not tick_size or tick_size <= 0:
            return round(float(price), 6)
        p = Decimal(str(price))
        t = Decimal(str(tick_size))
        q = (p / t).to_integral_value(rounding=ROUND_DOWN) * t
        return float(q)
    except Exception:
        return round(float(price), 6)

# =========================
# Veri ve Özellik Çıkarma
# =========================
async def fetch_klines(symbol, interval=None, limit=300):
    interval = get_valid_interval(interval or DEFAULT_INTERVAL)
    client = await init_binance_client()
    try:
        if not client:
            return []
        if hasattr(client, "futures_klines"):
            klines = await client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        else:
            klines = await client.fapiPublic_get_klines(symbol=symbol, interval=interval, limit=limit)
        return [{
            'timestamp': datetime.fromtimestamp(k[0]/1000) if len(k) > 0 else None,
            'open': float(k[1]) if len(k) > 1 else 0.0,
            'high': float(k[2]) if len(k) > 2 else 0.0,
            'low': float(k[3]) if len(k) > 3 else 0.0,
            'close': float(k[4]) if len(k) > 4 else 0.0,
            'volume': float(k[5]) if len(k) > 5 else 0.0
        } for k in klines if len(k) >= 6]
    except Exception as e:
        logger.error(f"{symbol} için fetch_klines hatası: {e}")
        return []
    finally:
        if client:
            await client.close_connection()

async def fetch_liquidity_data(symbol):
    client = None
    try:
        client = await init_binance_client()
        if not client:
            return {'volume_24h': 0, 'price_change': 0}
        try:
            if hasattr(client, "futures_ticker"):
                ticker = await asyncio.wait_for(client.futures_ticker(symbol=symbol), timeout=API_TIMEOUT)
            else:
                ticker = await asyncio.wait_for(client.fapiPublic_get_ticker_24hr(symbol=symbol), timeout=API_TIMEOUT)
        except AttributeError:
            ticker = await asyncio.wait_for(client.fapiPublic_get_ticker_24hr(symbol=symbol), timeout=API_TIMEOUT)
        if isinstance(ticker, list) and ticker:
            ticker = ticker[0]
        return {
            'volume_24h': float(ticker.get('quoteVolume', 0.0)),
            'price_change': float(ticker.get('priceChangePercent', 0.0))
        }
    except asyncio.TimeoutError:
        logger.error(f"{symbol} için likidite verisi timeout!")
        return {'volume_24h': 0, 'price_change': 0}
    except Exception as e:
        logger.error(f"Likidite verisi alınamadı {symbol}: {e}")
        return {'volume_24h': 0, 'price_change': 0}
    finally:
        if client:
            await client.close_connection()

async def fetch_symbols():
    cached = load_symbol_cache()
    if cached:
        return [s for s in cached if s.endswith("USDT")]
    try:
        client = await init_binance_client()
        if not client:
            return []
        exchange_info = await asyncio.wait_for(client.futures_exchange_info(), timeout=API_TIMEOUT)
        symbols = [
            s["symbol"] for s in exchange_info["symbols"]
            if s["contractType"] == "PERPETUAL" and s["symbol"].endswith("USDT")
        ]
        await client.close_connection()
        blacklist = load_blacklist()
        symbols = [s for s in symbols if s not in blacklist]
        save_symbol_cache(symbols)
        logger.debug(f"fetch_symbols: {len(symbols)} sembol bulundu")
        return symbols
    except asyncio.TimeoutError:
        logger.error("fetch_symbols: Timeout oluştu - semboller alınamadı.")
        return []
    except Exception as e:
        logger.error(f"Binance sembolleri alınamadı: {e}")
        return []

async def build_features_dataframe(symbols: List[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        klines = await fetch_klines(symbol, limit=100)
        if not klines or len(klines) < 30:
            continue
        close = np.array([k['close'] for k in klines], dtype=np.float64)
        high = np.array([k['high'] for k in klines], dtype=np.float64)
        low = np.array([k['low'] for k in klines], dtype=np.float64)
        volume = np.array([k['volume'] for k in klines], dtype=np.float64)
        try:
            rsi = talib.RSI(close, 14)[-1]
            ema = talib.EMA(close, 20)[-1]
            macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            macd_direction = 1 if macd[-1] > macd_signal[-1] else 0
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100 if (bb_upper[-1] - bb_lower[-1]) != 0 else 50
            atr = talib.ATR(high, low, close, 14)[-1]
            atr_percent = atr / close[-1] * 100 if close[-1] > 0 else 0
            volume_ratio = volume[-1] / np.mean(volume[-20:]) if np.mean(volume[-20:]) > 0 else 0
            ema_diff = ((close[-1] - ema) / ema) * 100 if ema > 0 else 0
        except Exception as e:
            logger.warning(f"{symbol}: Teknik gösterge hesaplama hatası: {e}")
            continue
        row = {
            "symbol": symbol,
            "signal_strength": 3,
            "rsi": float(rsi),
            "ema_diff": float(ema_diff),
            "macd_direction": int(macd_direction),
            "bb_position": float(bb_position),
            "volume_ratio": float(volume_ratio),
            "atr_percent": float(atr_percent),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# =========================
# Pozisyon/History IO
# =========================
def load_positions():
    if not os.path.exists(POSITION_FILE):
        return []
    with open(POSITION_FILE, "r") as f:
        return json.load(f)

def save_positions(positions):
    atomic_write_json(POSITION_FILE, convert_numpy(positions))

def record_closed_trade(pos, exit_price, reason):
    entry = pos["entry_price"]
    size = pos["size"]
    side = pos["side"]
    if side == "long":
        pnl_percent = ((exit_price - entry) / entry) * 100
    else:
        pnl_percent = ((entry - exit_price) / entry) * 100
    profit_usdt = (pnl_percent / 100.0) * size

    # R hesapla
    sl_price = pos.get("sl")
    risk_per_unit = None
    R_multiple = None
    if sl_price and entry and entry > 0 and sl_price > 0:
        if side == "long":
            risk_per_unit = entry - sl_price
            move = exit_price - entry
        else:
            risk_per_unit = sl_price - entry
            move = entry - exit_price
        if risk_per_unit and risk_per_unit > 0:
            R_multiple = move / risk_per_unit

    global risk_manager
    if risk_manager:
        risk_manager.record_trade(profit_usdt)

    trade = {
        "symbol": pos["symbol"],
        "side": side,
        "entry_price": entry,
        "exit_price": exit_price,
        "pnl_percent": round(pnl_percent, 2),
        "size": size,
        "profit_usdt": round(profit_usdt, 4),
        "closed_reason": reason,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_strength": pos.get("signal_strength", 0),
        "rsi": pos.get("rsi", 0),
        "ml_probability": pos.get("ml_probability", 0),
        "mode": pos.get("mode", "real"),
        "tp1_hit": pos.get("tp1_hit", False),
        "tp2_hit": pos.get("tp2_hit", False),
        "r": R_multiple
    }
    for feature in REQUIRED_FEATURES:
        trade[feature] = pos.get(feature, 0)
    pos['closed'] = True

    if not os.path.exists(HISTORY_FILE):
        atomic_write_json(HISTORY_FILE, [trade])
    else:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        history.append(trade)
        atomic_write_json(HISTORY_FILE, history)

    # Adaptif kaydı
    if 'adaptive_manager' in globals() and adaptive_manager and adaptive_manager.enabled:
        try:
            adaptive_manager.record_trade(trade)
        except Exception as e:
            logger.error(f"Adaptive record_trade hata: {e}")

async def batch_predict_symbols(symbols_data):
    """Birçok sembol için ML tahminlerini toplu olarak yapar"""
    if not model_cls:
        return {}
        
    features_list = []
    symbols = []
    
    for symbol, features in symbols_data.items():
        try:
            X = select_features_frame(features)
            features_list.append(X)
            symbols.append(symbol)
        except Exception as e:
            logger.error(f"Batch ML hata {symbol}: {e}")
    
    if not features_list:
        return {}
        
    # Tüm özellikleri birleştir
    X_batch = pd.concat(features_list, ignore_index=True)
    
    try:
        # Toplu tahmin
        probabilities = model_cls.predict_proba(X_batch)[:, 1]
        
        if is_invert_prob():
            probabilities = 1.0 - probabilities
            
        # Sonuçları eşle
        results = {}
        for i, symbol in enumerate(symbols):
            if i < len(probabilities):
                results[symbol] = float(probabilities[i])
        
        return results
    except Exception as e:
        logger.error(f"Batch ML tahmin hatası: {e}")
        return {}

# =========================
# Risk ve TP/SL
# =========================
class RiskCalculator:
    @staticmethod
    def calculate_position_size(symbol: str, atr: float, current_price: float, account_balance: float) -> float:
        risk_amount = account_balance * MAX_ACCOUNT_RISK_PERCENT / 100.0
        try:
            size = risk_amount / max(atr * 2, 1e-6)
        except Exception:
            size = risk_amount / 2
        return min(size * current_price, account_balance * 0.1)

    @staticmethod
    def position_notional(account_balance: float, atr: float, current_price: float) -> float:
        if TARGET_MARGIN_USDT > 0:
            notional = TARGET_MARGIN_USDT * DEFAULT_LEVERAGE
        elif POSITION_SIZING_MODE == "percent":
            notional = account_balance * (POSITION_PERCENT / 100.0)
        else:
            return RiskCalculator.calculate_position_size("N/A", atr, current_price, account_balance)
        
        notional = max(notional, MIN_NOTIONAL_USDT)
        notional = min(notional, account_balance * (MAX_NOTIONAL_PERCENT / 100.0))
        return notional

def calc_tp_sl_abs(entry_price: float, atr: float, probability: float, side: str, tick_size: float) -> Tuple[float,float,float]:
    tp1_multiplier = float(os.getenv("ATR_TP1_MULTIPLIER", "1.5"))
    tp2_multiplier = float(os.getenv("ATR_TP2_MULTIPLIER", "3.0"))
    sl_multiplier  = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))

    if 'adaptive_manager' in globals() and adaptive_manager and adaptive_manager.enabled:
        ap = adaptive_manager.get_runtime_params()
        tp1_multiplier = ap.get("atr_tp1_multiplier", tp1_multiplier)
        tp2_multiplier = ap.get("atr_tp2_multiplier", tp2_multiplier)
        sl_multiplier  = ap.get("atr_sl_multiplier",  sl_multiplier)
    
    base_multiplier = 1 + (probability * 0.5)
    tp1_dist = atr * tp1_multiplier * base_multiplier
    tp2_dist = atr * tp2_multiplier * base_multiplier
    sl_dist = atr * sl_multiplier * base_multiplier
    
    if side == "long":
        tp1 = entry_price + tp1_dist
        tp2 = entry_price + tp2_dist
        sl = entry_price - sl_dist
    else:
        tp1 = entry_price - tp1_dist
        tp2 = entry_price - tp2_dist
        sl = entry_price + sl_dist
    
    tp1 = adjust_price_to_tick(tp1, tick_size)
    tp2 = adjust_price_to_tick(tp2, tick_size)
    sl = adjust_price_to_tick(sl, tick_size)
    
    return tp1, tp2, sl

# =========================
# Binance Emir Fonksiyonları
# =========================
async def get_futures_balance():
    client = await init_binance_client()
    try:
        if not client:
            return 0.0
        account_info = await client.futures_account()
        for asset in account_info.get('assets', []):
            if asset.get('asset') == 'USDT':
                return float(asset.get('availableBalance', 0.0))
        return 0.0
    except Exception as e:
        logger.error(f"Bakiye sorgulama hatası: {e}")
        return 0.0
    finally:
        if client:
            await client.close_connection()

async def open_binance_position(symbol, side, quantity):
    client = await init_binance_client()
    try:
        if not client:
            return None
        order = await client.futures_create_order(
            symbol=symbol,
            side="BUY" if side == "long" else "SELL",
            type="MARKET",
            quantity=quantity
        )
        logger.info(f"Gerçek emir açıldı: {order}")
        return order
    except Exception as e:
        logger.error(f"Binance order açma hatası: {e}")
        return None
    finally:
        if client:
            await client.close_connection()

async def get_top_volume_symbols(client, limit=30):
    """En yüksek işlem hacimli coinleri getirir"""
    try:
        # Son 24 saatin hacim verilerini al
        tickers = await client.futures_ticker()
        
        # İşlem hacmine göre sırala
        sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
        
        # İlk 'limit' kadar coin'i döndür
        return [ticker['symbol'] for ticker in sorted_tickers[:limit] if ticker['symbol'].endswith('USDT')]
    except Exception as e:
        logger.error(f"Volume verisi alınamadı: {e}")
        return []

async def get_symbol_quantity_precision_and_step(symbol):
    client = await init_binance_client()
    try:
        if not client:
            return 3, 0.001
        info = await client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = float(f['stepSize'])
                        precision = abs(Decimal(str(step_size)).as_tuple().exponent)
                        return precision, step_size
        return 3, 0.001
    except Exception as e:
        logger.error(f"Precision çekme hatası: {e}")
        return 3, 0.001
    finally:
        if client:
            await client.close_connection()

async def set_leverage(symbol, leverage):
    client = await init_binance_client()
    try:
        if not client:
            return
        await client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logger.info(f"{symbol}: Kaldıraç {leverage} olarak ayarlandı.")
    except Exception as e:
        logger.error(f"Kaldıraç ayarlanamadı ({symbol}): {e}")
    finally:
        if client:
            await client.close_connection()

def adjust_quantity(quantity, precision, step_size):
    quant = Decimal(str(quantity))
    quant = quant.quantize(Decimal('1.' + '0' * int(abs(Decimal(str(step_size)).as_tuple().exponent))), rounding=ROUND_DOWN)
    steps = (quant // Decimal(str(step_size)))
    quant = Decimal(str(step_size)) * steps
    return float(quant)

async def open_tp_sl_orders(symbol, side, q_tp1, q_tp2, q_sl, tp1, tp2, sl, trailing_callback_rate: Optional[float] = None, activation_price: Optional[float] = None):
    client = await init_binance_client()
    close_side = "SELL" if side == "long" else "BUY"
    working_type = os.getenv("WORKING_PRICE_TYPE", "MARK_PRICE")
    use_price_protect = env_bool("PRICE_PROTECT", True)
    try:
        if not client:
            return
        if q_tp1 and q_tp1 > 0:
            await client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp1,
                quantity=q_tp1,
                reduceOnly=True,
                workingType=working_type,
                priceProtect=use_price_protect
            )
        if q_tp2 and q_tp2 > 0:
            await client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp2,
                quantity=q_tp2,
                reduceOnly=True,
                workingType=working_type,
                priceProtect=use_price_protect
            )
        if q_sl and q_sl > 0:
            await client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=sl,
                quantity=q_sl,
                reduceOnly=True,
                workingType=working_type,
                priceProtect=use_price_protect
            )
        if trailing_callback_rate and 0.1 <= trailing_callback_rate <= 5:
            await client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="TRAILING_STOP_MARKET",
                quantity=q_sl,
                activationPrice=activation_price if activation_price else tp1,
                callbackRate=round(trailing_callback_rate, 2),
                workingType=working_type,
                reduceOnly=True
            )
        logger.info(f"{symbol}: TP/SL ve Trailing emirleri başarıyla gönderildi.")
    except Exception as e:
        logger.error(f"{symbol}: TP/SL veya Trailing emirlerinde hata: {e}")
    finally:
        if client:
            await client.close_connection()

async def cancel_all_open_orders(symbol: str) -> bool:
    client = await init_binance_client()
    if not client:
        return False
    try:
        await client.futures_cancel_all_open_orders(symbol=symbol)
        logger.info(f"{symbol}: Tüm açık emirler iptal edildi.")
        return True
    except Exception as e:
        logger.error(f"{symbol}: Açık emirler iptal hatası: {e}")
        return False
    finally:
        await client.close_connection()

async def get_current_position(symbol: str):
    client = await init_binance_client()
    if not client:
        return 0.0, None, 0.0, 0.0
    try:
        account = await client.futures_account()
        for p in account.get("positions", []):
            if p.get("symbol") == symbol:
                amt = float(p.get("positionAmt") or 0.0)
                entry = float(p.get("entryPrice") or 0.0)
                mark = float(p.get("markPrice") or 0.0)
                qty = abs(amt)
                side = "long" if amt > 0 else ("short" if amt < 0 else None)
                return qty, side, entry, mark
        return 0.0, None, 0.0, 0.0
    except Exception as e:
        logger.error(f"{symbol}: Mevcut pozisyon çekilemedi: {e}")
        return 0.0, None, 0.0, 0.0
    finally:
        await client.close_connection()

async def count_open_by_side_exchange() -> Dict[str, int]:
    client = await init_binance_client()
    if not client:
        return {"long": 0, "short": 0}
    
    try:
        account = await client.futures_account()
        positions = account.get("positions", [])
        
        long_count = 0
        short_count = 0
        
        for pos in positions:
            qty = float(pos.get("positionAmt", 0))
            if abs(qty) > 1e-8:
                if qty > 0:
                    long_count += 1
                else:
                    short_count += 1
                    
        return {"long": long_count, "short": short_count}
        
    except Exception as e:
        logger.error(f"count_open_by_side_exchange hatası: {e}")
        return {"long": 0, "short": 0}
    finally:
        await client.close_connection()

async def infer_exchange_close_reason(symbol: str) -> Optional[str]:
    client = await init_binance_client()
    if not client:
        return None
    try:
        orders = await client.futures_get_all_orders(symbol=symbol, limit=50)
        if not orders:
            return None
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        recent = [
            o for o in orders
            if o.get("reduceOnly") and o.get("status") == "FILLED"
            and (now_ms - int(o.get("updateTime", now_ms))) < 30*60*1000
        ]
        if not recent:
            return None
        recent.sort(key=lambda o: int(o.get("updateTime", 0)), reverse=True)
        typ = recent[0].get("type")
        if typ == "TAKE_PROFIT_MARKET":
            return "TP"
        if typ == "STOP_MARKET":
            return "SL"
        if typ == "TRAILING_STOP_MARKET":
            return "Trailing"
        return None
    except Exception as e:
        logger.error(f"{symbol}: infer_exchange_close_reason hata: {e}")
        return None
    finally:
        try:
            await client.close_connection()
        except:
            pass

async def get_realized_pnl_sum(symbol: str, start_dt: Optional[datetime], end_dt: Optional[datetime] = None) -> Optional[float]:
    client = await init_binance_client()
    if not client:
        return None
    try:
        start_ms = int((start_dt or (datetime.utcnow() - timedelta(days=1))).timestamp() * 1000)
        end_ms = int((end_dt or datetime.utcnow()).timestamp() * 1000)
        items = None
        try:
            items = await client.futures_income_history(
                symbol=symbol, incomeType="REALIZED_PNL", startTime=start_ms, endTime=end_ms
            )
        except AttributeError:
            items = await client.fapiPrivate_get_income(
                symbol=symbol, incomeType="REALIZED_PNL", startTime=start_ms, endTime=end_ms
            )
        if not items:
            return None
        total = 0.0
        for it in items:
            try:
                total += float(it.get("income", 0.0))
            except Exception:
                continue
        return round(total, 6)
    except Exception as e:
        logger.error(f"{symbol}: get_realized_pnl_sum hata: {e}")
        return None
    finally:
        try:
            await client.close_connection()
        except:
            pass

def compute_approx_pnl(pos: dict, exit_price: float) -> Tuple[float, float]:
    try:
        entry = float(pos.get("entry_price", 0.0) or 0.0)
        side = (pos.get("side") or "long").lower()
        size_usdt = float(pos.get("size", 0.0) or 0.0)
        if entry <= 0 or size_usdt <= 0:
            return 0.0, 0.0
        if side == "long":
            pnl_pct = ((exit_price - entry) / entry) * 100.0
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100.0
        profit_usdt = (pnl_pct / 100.0) * size_usdt
        return round(pnl_pct, 2), round(profit_usdt, 6)
    except Exception:
        return 0.0, 0.0

async def close_position_market(symbol: str, side: str, quantity: float) -> bool:
    if quantity <= 0:
        logger.warning(f"{symbol}: Kapatılacak miktar 0 görünüyor.")
        return False
    client = await init_binance_client()
    if not client:
        return False
    try:
        precision, step = await get_symbol_quantity_precision_and_step(symbol)
        qty_adj = adjust_quantity(quantity, precision, step)
        close_side = "SELL" if side == "long" else "BUY"
        await client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type="MARKET",
            quantity=qty_adj,
            reduceOnly=True
        )
        logger.info(f"{symbol}: Pozisyon MARKET (reduceOnly) ile kapatıldı. qty={qty_adj}")
        return True
    except Exception as e:
        logger.error(f"{symbol}: MARKET kapatma hatası: {e}")
        return False
    finally:
        await client.close_connection()

async def get_open_orders(symbol: str):
    client = await init_binance_client()
    if not client:
        return []
    try:
        orders = await client.futures_get_open_orders(symbol=symbol)
        return orders or []
    except Exception as e:
        logger.error(f"{symbol}: Açık emirler alınamadı: {e}")
        return []
    finally:
        await client.close_connection()

async def replace_stop_loss_order(symbol: str, side: str, new_qty: float, sl_price: float) -> bool:
    if new_qty <= 0 or not sl_price:
        return False

    client = await init_binance_client()
    if not client:
        return False
    working_type = os.getenv("WORKING_PRICE_TYPE", "MARK_PRICE")
    use_price_protect = env_bool("PRICE_PROTECT", True)
    try:
        existing = await client.futures_get_open_orders(symbol=symbol)
        stop_orders = [o for o in (existing or []) if o.get("type") == "STOP_MARKET"]

        for o in stop_orders:
            try:
                await client.futures_cancel_order(symbol=symbol, orderId=o.get("orderId"))
            except Exception as ce:
                logger.warning(f"{symbol}: SL emri iptal edilemedi ({o.get('orderId')}): {ce}")

        precision, step = await get_symbol_quantity_precision_and_step(symbol)
        qty_adj = adjust_quantity(new_qty, precision, step)

        close_side = "SELL" if side == "long" else "BUY"
        await client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type="STOP_MARKET",
            stopPrice=sl_price,
            quantity=qty_adj,
            reduceOnly=True,
            workingType=working_type,
            priceProtect=use_price_protect
        )
        logger.info(f"{symbol}: SL {qty_adj} miktarla yenilendi @ {sl_price}")
        return True
    except Exception as e:
        logger.error(f"{symbol}: SL yenileme hatası: {e}")
        return False
    finally:
        await client.close_connection()

# =========================
# Sinyal ve Strateji
# =========================
def log_shadow_trade(symbol, side, signal_strength, rsi, probability, features):
    try:
        if features.get('volume_ratio', 1) == 0:
            add_to_blacklist(symbol)
            logger.warning(f"Volume ratio 0 tespit edildi! {symbol} blacklist'e eklendi")
            return
        if 'atr_percent' in features and features['atr_percent'] > 2.0:
            features['atr_percent'] = 2.0
            logger.warning(f"ATR düzeltildi ({symbol}) -> 2.0%")
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        else:
            history = []
        trade = {
            "symbol": symbol,
            "side": side,
            "entry_price": 0,
            "exit_price": 0,
            "pnl_percent": 0,
            "size": 0,
            "profit_usdt": 0,
            "closed_reason": "shadow",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "signal_strength": signal_strength,
            "rsi": rsi,
            "ml_probability": probability,
            "mode": "shadow",
            **features
        }
        history.append(trade)
        atomic_write_json(HISTORY_FILE, history[-10_000:])
        logger.info(f"👻 Shadow verisi kaydedildi: {symbol}")
    except Exception as e:
        logger.error(f"Shadow veri kaydetme hatası: {e}")

async def get_htf_trend(symbol: str) -> Optional[str]:
    if not env_bool("USE_HTF_TREND", False):
        return None
    tf = os.getenv("HTF_TREND_TF", "1h")
    ema_len = int(os.getenv("HTF_TREND_EMA", "200"))
    tol_bp = float(os.getenv("HTF_TREND_TOL_BP", "5.0"))
    kl = await fetch_klines(symbol, interval=tf, limit=max(ema_len + 10, 250))
    if not kl or len(kl) < ema_len + 5:
        return None
    closes = np.array([k["close"] for k in kl], dtype=np.float64)
    try:
        ema_series = talib.EMA(closes, ema_len)
    except Exception:
        return None
    if ema_series is None or np.isnan(ema_series[-1]):
        return None
    ema_val = float(ema_series[-1])
    last_close = float(closes[-1])
    if ema_val <= 0:
        return None
    tol = tol_bp / 10000.0
    up = last_close >= ema_val * (1.0 - tol)
    down = last_close <= ema_val * (1.0 + tol)
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return "none"

async def check_signal(symbol, rsi, ema, close_prices, volume, open_prices, high_prices, low_prices, macd, macd_signal, upper_bb, lower_bb, custom_params=None):
    try:
        custom_params = custom_params or {}
        min_volume = float(os.getenv("MIN_VOLUME_RATIO", 0.8))
        min_price = float(os.getenv("MIN_PRICE", 0.05))

        if (
            len(close_prices) < 20
            or len(volume) < 20
            or not symbol.endswith("USDT")
        ):
            return None, 0, {}, 0.0

        last_close = float(close_prices[-1])
        if last_close < min_price:
            return None, 0, {}, 0.0

        # Göstergelerin son değerleri güvenli alınır
        if len(rsi) == 0 or len(ema) == 0 or len(macd) == 0 or len(macd_signal) == 0 or len(upper_bb) == 0 or len(lower_bb) == 0:
            return None, 0, {}, 0.0

        last_ema = float(ema[-1])
        last_rsi = float(rsi[-1])
        last_macd = float(macd[-1])
        last_macd_signal = float(macd_signal[-1])

        # ATR / volatilite
        try:
            atr_val = talib.ATR(
                np.asarray(high_prices, dtype=float),
                np.asarray(low_prices, dtype=float),
                np.asarray(close_prices, dtype=float),
                14
            )[-1]
            atr = float(atr_val) if np.isfinite(atr_val) else 0.0
        except Exception:
            atr = 0.0

        volatility_ratio = (atr / last_close) if last_close > 0 else 0.0

        # ADAPTIF PARAMETRELER (önce temel değerleri türet)
        # Trend / Range durumuna göre temel RSI bandı (volatility_ratio > 0.02 ise trend kabul edelim örnek)
        if volatility_ratio > 0.02:
            base_rsi_min = float(os.getenv("TREND_RSI_MIN", 25))
            base_rsi_max = float(os.getenv("TREND_RSI_MAX", 75))
        else:
            base_rsi_min = float(os.getenv("RANGE_RSI_MIN", 40))
            base_rsi_max = float(os.getenv("RANGE_RSI_MAX", 60))

        # short için temel band (fallback env)
        base_short_rsi_min = float(os.getenv("SHORT_RSI_MIN", 40))
        base_short_rsi_max = float(os.getenv("SHORT_RSI_MAX", 80))

        adaptive_params = {}
        if 'adaptive_manager' in globals() and adaptive_manager and adaptive_manager.enabled:
            adaptive_params = adaptive_manager.get_runtime_params()

        # Adaptif override (varsa)
        final_rsi_min = adaptive_params.get("rsi_low", custom_params.get("rsi_min", base_rsi_min))
        final_rsi_max = adaptive_params.get("rsi_high", custom_params.get("rsi_max", base_rsi_max))

        # Güvenli sınır (mantıksızlık önle)
        if final_rsi_min >= final_rsi_max:  # yanlış bir profil gelirse düzelt
            final_rsi_min, final_rsi_max = sorted([final_rsi_min, final_rsi_max])

        short_rsi_min = adaptive_params.get("short_rsi_low", float(os.getenv("SHORT_RSI_MIN", base_short_rsi_min)))
        short_rsi_max = adaptive_params.get("short_rsi_high", float(os.getenv("SHORT_RSI_MAX", base_short_rsi_max)))
        if short_rsi_min >= short_rsi_max:
            short_rsi_min, short_rsi_max = sorted([short_rsi_min, short_rsi_max])

        avg_volume = float(np.mean(volume[-20:])) if len(volume) >= 20 else 0.0
        volume_ratio = (float(volume[-1]) / avg_volume) if avg_volume > 0 else 0.0
        if volume_ratio < min_volume:
            logger.debug(f"{symbol}: Volume ratio {volume_ratio:.2f} < min {min_volume:.2f}")
            return None, 0, {}, 0.0

        # RSI band dışı ise direkt eler
        if last_rsi > final_rsi_max or last_rsi < final_rsi_min:
            logger.debug(f"{symbol}: RSI {last_rsi:.1f} outside range [{final_rsi_min:.1f}-{final_rsi_max:.1f}]")
            return None, 0, {}, 0.0

        # Bollinger konumu
        try:
            bb_range = float(upper_bb[-1]) - float(lower_bb[-1])
            bb_position = ((last_close - float(lower_bb[-1])) / bb_range) * 100 if bb_range != 0 else 50.0
        except Exception:
            bb_position = 50.0

        macd_above_signal = last_macd > last_macd_signal
        macd_cross = False
        if len(macd) >= 2 and len(macd_signal) >= 2:
            prev_macd = float(macd[-2])
            prev_signal = float(macd_signal[-2])
            macd_cross = (last_macd > last_macd_signal) and (prev_macd <= prev_signal)

        ema_diff = ((last_close - last_ema) / last_ema) * 100.0 if last_ema > 0 else 0.0
        market_condition = "trend" if abs(last_ema - last_close) > (atr * 0.5) else "range"

        features = {
            "signal_strength": 3,
            "rsi": float(last_rsi),
            "atr_percent": float(volatility_ratio * 100.0),
            "volume_ratio": float(volume_ratio),
            "ema_diff": float(ema_diff),
            "macd_direction": 1 if macd_above_signal else 0,
            "macd_cross": 1 if macd_cross else 0,
            "bb_position": float(bb_position),
            "price_ema_ratio": float(last_close / last_ema) if last_ema > 0 else 1.0,
            "market_condition": market_condition,
        }

        # Feature engineering (ML ile tutarlılık)
        if "bb_position" in features:
            features["bb_position_squared"] = features["bb_position"] ** 2
        if "rsi" in features:
            features["rsi_normalized"] = (features["rsi"] - 50) / 50
        if all(k in features for k in ["bb_position","rsi","rsi_normalized"]):
            features["bb_rsi_ratio"] = features["bb_position"] * features["rsi_normalized"]
        if "volume_ratio" in features and "atr_percent" in features:
            features["volume_volatility_ratio"] = features["volume_ratio"] / (features["atr_percent"] + 0.1)
        if "ema_diff" in features and "macd_direction" in features:
            features["trend_strength"] = features["ema_diff"] * features["macd_direction"]

        # ML Olasılık
        probability = 0.7
        if model_cls:
            try:
                X = select_features_frame(features)
                probability = float(model_cls.predict_proba(X)[0][1])
                if is_invert_prob():
                    probability = 1.0 - probability
                features["signal_strength"] = min(5, int(probability * 10))
            except Exception as ml_e:
                logger.error(f"ML prediction error ({symbol}): {ml_e}")

        thr = custom_params.get("ml_threshold", get_dynamic_ml_threshold(symbol))
        logger.debug(f"{symbol}: ML olasılık={probability:.3f}, dinamik eşik={thr:.3f}")

        ml_min_margin = float(os.getenv("ML_MIN_MARGIN", "0.0"))
        if ml_min_margin > 0 and (probability - thr) < ml_min_margin:
            logger.debug(f"{symbol}: ML margin yetersiz (p={probability:.3f}, thr={thr:.3f}, min_margin={ml_min_margin:.3f})")
            return None, 0, {}, 0.0

        # Regresyon
        if model_reg:
            try:
                Xr = select_features_frame(features)
                reg_prediction = float(model_reg.predict(Xr)[0])
                features["reg_prediction"] = reg_prediction
                if FILTER_NEG_REGRESSION and reg_prediction < 0:
                    logger.debug(f"{symbol}: Negatif regresyon tahmini: {reg_prediction:.4f}, filtrelendi")
                    return None, 0, {}, 0.0
                if env_bool("REG_FORCE_POSITIVE", False) and reg_prediction < 0:
                    logger.debug(f"{symbol}: Regresyon negatif (force positive), filtrelendi")
                    return None, 0, {}, 0.0
            except Exception as re:
                logger.error(f"Regresyon tahmini hatası ({symbol}): {re}")

        # HTF trend
        trend = await get_htf_trend(symbol)
        if trend is not None:
            if trend == "up":
                htf_allows_long, htf_allows_short = True, False
            elif trend == "down":
                htf_allows_long, htf_allows_short = False, True
            else:
                htf_allows_long = htf_allows_short = False
        else:
            htf_allows_long = htf_allows_short = True

        favor_longs = custom_params.get("favor_long", True)
        favor_shorts = custom_params.get("favor_short", True)

        side = None
        if last_close > last_ema and htf_allows_long and favor_longs:
            if (macd_above_signal and bb_position > 30 and
                (probability > thr or (macd_cross and last_rsi < final_rsi_max - 10))):
                side = "long"
        elif last_close <= last_ema and htf_allows_short and favor_shorts:
            if not (last_rsi > short_rsi_max or last_rsi < short_rsi_min):
                if ((not macd_above_signal) and bb_position < 70 and
                    (probability > thr or (macd_cross and last_rsi > final_rsi_min + 10))):
                    side = "short"

        if not side:
            log_shadow_trade(
                symbol,
                "long" if last_close > last_ema else "short",
                float(features.get("signal_strength", 0.0)),
                float(last_rsi),
                float(probability),
                {**features, "htf_trend": trend, "rsi_band": (final_rsi_min, final_rsi_max)}
            )
            return None, 0, {}, 0.0

        # Yumuşak filtre
        ok_soft, reason_soft = await entry_soft_filters(symbol, side, float(last_close))
        if not ok_soft:
            log_shadow_trade(
                symbol,
                side,
                float(features.get("signal_strength", 0.0)),
                float(last_rsi),
                float(probability),
                {**features, "soft_filter_reason": reason_soft}
            )
            return None, 0, {}, 0.0

        # Pullback kontrolü
        ok_pullback, pullback_reason = await check_pullback_entry(symbol, side, float(last_close))
        if not ok_pullback:
            logger.info(f"{symbol}: Pullback kontrolü başarısız: {pullback_reason}")
            return None, 0, {}, 0.0

        # Adaptif ATR multipliers
        tp1_mult = adaptive_params.get("atr_tp1_multiplier", float(os.getenv("ATR_TP1_MULTIPLIER", "2.5")))
        tp2_mult = adaptive_params.get("atr_tp2_multiplier", float(os.getenv("ATR_TP2_MULTIPLIER", "4.0")))
        sl_mult  = adaptive_params.get("atr_sl_multiplier",  float(os.getenv("ATR_SL_MULTIPLIER", "1.8")))
        features["tp1_multiplier"] = tp1_mult
        features["tp2_multiplier"] = tp2_mult
        features["sl_multiplier"]  = sl_mult

        return side, float(features["signal_strength"]), features, float(probability)

    except Exception as e:
        logger.error(f"Signal processing error: {symbol} - {str(e)}", exc_info=True)
        return None, 0, {}, 0.0

async def enhanced_position_filter(symbol, side, probability, features) -> tuple[bool, str]:
    """Gelişmiş pozisyon filtresi"""
    
    # 1. Minimum sinyal gücü
    signal_strength = features.get("signal_strength", 0)
    if signal_strength < MIN_SIGNAL_STRENGTH:
        return False, f"weak_signal({signal_strength}<{MIN_SIGNAL_STRENGTH})"
    
    # 2. Volatilite kontrolü
    atr_pct = features.get("atr_percent", 0)
    min_atr = float(os.getenv("MIN_ATR_PERCENT", "0.5"))
    max_atr = float(os.getenv("MAX_ATR_PERCENT", "8.0"))
    
    if atr_pct < min_atr:
        return False, f"low_volatility({atr_pct:.2f}%<{min_atr}%)"
    
    if atr_pct > max_atr:
        return False, f"high_volatility({atr_pct:.2f}%>{max_atr}%)"
    
    # 3. Volume momentum kontrolü
    volume_ratio = features.get("volume_ratio", 0)
    min_volume_momentum = float(os.getenv("MIN_VOLUME_MOMENTUM", "1.2"))
    
    if volume_ratio < min_volume_momentum:
        return False, f"low_volume_momentum({volume_ratio:.2f}<{min_volume_momentum})"
    
    # 4. ML güven aralığı
    ml_margin = probability - get_dynamic_ml_threshold(symbol)
    min_ml_margin = float(os.getenv("MIN_ML_CONFIDENCE_MARGIN", "0.05"))
    
    if ml_margin < min_ml_margin:
        return False, f"low_ml_margin({ml_margin:.3f}<{min_ml_margin})"
    
    # 5. Komisyon karlılık tahmini
    estimated_profit_pct = probability * 3.0  # Tahmini kar %3
    commission_cost_pct = COMMISSION_RATE * 2 * 100  # Toplam komisyon
    
    if estimated_profit_pct < commission_cost_pct * 2:  # 2x komisyon
        return False, f"low_profit_potential({estimated_profit_pct:.2f}%<{commission_cost_pct*2:.2f}%)"
    
    return True, "all_filters_passed"

def check_risk_reward_ratio(entry_price: float, tp1: float, sl: float, side: str) -> bool:
    """Minimum risk/reward oranı kontrolü"""
    
    if side == "long":
        potential_profit = tp1 - entry_price
        potential_loss = entry_price - sl
    else:
        potential_profit = entry_price - tp1
        potential_loss = sl - entry_price
    
    if potential_loss <= 0:
        return False
        
    risk_reward_ratio = potential_profit / potential_loss
    min_ratio = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5"))
    
    return risk_reward_ratio >= min_ratio

def calc_dynamic_tp_sl(entry_price: float, atr: float, ml_probability: float, side: str) -> tuple:
    """ML olasılığına göre dinamik TP/SL"""
    
    # Yüksek olasılık = daha agresif kar alma
    if ml_probability > 0.80:
        tp1_mult = 1.2  # Çok yakın kar al
        tp2_mult = 2.5
        sl_mult = 2.5   # SL uzak
    elif ml_probability > 0.65:
        tp1_mult = 1.5
        tp2_mult = 3.0
        sl_mult = 2.0
    else:
        tp1_mult = 1.8
        tp2_mult = 3.5
        sl_mult = 1.8   # SL yakın (düşük güven)
    
    if side == "long":
        tp1 = entry_price + (atr * tp1_mult)
        tp2 = entry_price + (atr * tp2_mult)
        sl = entry_price - (atr * sl_mult)
    else:
        tp1 = entry_price - (atr * tp1_mult)
        tp2 = entry_price - (atr * tp2_mult)
        sl = entry_price + (atr * sl_mult)
    
    return tp1, tp2, sl

def check_sufficient_movement_potential(symbol, current_price, tp1, sl):
    """Yeterli hareket potansiyeli var mı?"""
    
    # Minimum pip mesafesi
    min_tp_pips = float(os.getenv("MIN_TP_PIPS", "50"))  # 50 pip
    min_sl_pips = float(os.getenv("MIN_SL_PIPS", "25"))  # 25 pip
    
    # Pip hesaplama (USDT çiftleri için)
    pip_size = 0.0001 if "USDT" in symbol else 0.00001
    
    tp_pips = abs(tp1 - current_price) / pip_size
    sl_pips = abs(current_price - sl) / pip_size
    
    if tp_pips < min_tp_pips:
        logger.info(f"{symbol}: TP mesafesi çok yakın ({tp_pips:.0f} < {min_tp_pips} pip)")
        return False
        
    if sl_pips < min_sl_pips:
        logger.info(f"{symbol}: SL mesafesi çok yakın ({sl_pips:.0f} < {min_sl_pips} pip)")
        return False
    
    return True

async def wait_for_limit_fill(symbol: str, order_id: int) -> tuple[Optional[float], float, float]:
    """
    Limit emrin fill olmasını (veya kısmen fill) bekler.
    Döner: (avg_price or None, executed_qty, orig_qty)
    """
    client = await init_binance_client()
    if not client:
        return None, 0.0, 0.0
    start = time.time()
    last_status = None
    try:
        while (time.time() - start) <= LIMIT_FILL_TIMEOUT_SEC:
            try:
                order = await client.futures_get_order(symbol=symbol, orderId=order_id)
            except Exception as e:
                logger.error(f"{symbol}: futures_get_order hata: {e}")
                await asyncio.sleep(1)
                continue
            status = order.get("status")
            orig_qty = float(order.get("origQty", 0))
            exec_qty = float(order.get("executedQty", 0))
            avg_price = float(order.get("avgPrice", 0)) if exec_qty > 0 else None

            if status != last_status:
                logger.debug(f"{symbol}: Order status={status}, exec={exec_qty}/{orig_qty}")
                last_status = status

            # Tamamen doldu
            if status == "FILLED" and exec_qty > 0:
                return avg_price, exec_qty, orig_qty

            # Kısmi dolum (status genelde PARTIALLY_FILLED veya NEW ama execQty>0)
            if exec_qty > 0 and orig_qty > 0:
                fill_ratio = exec_qty / orig_qty
                if fill_ratio >= MIN_FILLED_RATIO_FOR_ACTIVATION:
                    return avg_price, exec_qty, orig_qty

            await asyncio.sleep(1)

        # Timeout
        if CANCEL_UNFILLED_LIMIT:
            try:
                await client.futures_cancel_order(symbol=symbol, orderId=order_id)
                logger.info(f"{symbol}: Limit emir timeout → iptal edildi.")
            except Exception as ce:
                logger.warning(f"{symbol}: Timeout sonrası iptal başarısız: {ce}")
        return None, 0.0, 0.0
    finally:
        try:
            await client.close_connection()
        except:
            pass

async def finalize_filled_position(positions: list, symbol: str, side: str,
                                   filled_price: float, exec_qty: float,
                                   strength: float, features: dict, probability: float):
    """
    Limit emri dolduktan (tam veya kısmi) sonra gerçek pozisyonu oluşturur
    ve TP/SL emirlerini yerleştirir.
    """
    if exec_qty <= 0 or filled_price <= 0:
        logger.warning(f"{symbol}: finalize_filled_position geçersiz exec_qty/price")
        return

    precision, step_size, min_qty, ex_min_notional, tick_size = await get_symbol_trading_filters(symbol)

    # ATR hesapları için kline çek
    kl = await fetch_klines(symbol, limit=50)
    if len(kl) < 20:
        logger.warning(f"{symbol}: finalize sırasında ATR için yetersiz kline.")
        return
    close = np.array([k['close'] for k in kl], dtype=np.float64)
    high = np.array([k['high'] for k in kl], dtype=np.float64)
    low  = np.array([k['low'] for k in kl],  dtype=np.float64)
    atr_val = float(talib.ATR(high, low, close, 14)[-1])

    notional = exec_qty * filled_price
    tp1, tp2, sl = calc_tp_sl_abs(filled_price, atr_val, probability, side, tick_size)

    q_tp1_frac = env_float("TP1_FRACTION", 0.30)
    q_tp1 = adjust_quantity(exec_qty * q_tp1_frac, precision, step_size)
    q_tp2 = adjust_quantity(max(exec_qty - q_tp1, 0.0), precision, step_size)
    q_sl = exec_qty

    trailing_callback_rate = max(0.1, min(5.0, TRAILING_OFFSET_ENV))
    activation_price = adjust_price_to_tick(tp2, tick_size)
    await open_tp_sl_orders(
        symbol, side, q_tp1, q_tp2, q_sl,
        adjust_price_to_tick(tp1, tick_size),
        adjust_price_to_tick(tp2, tick_size),
        adjust_price_to_tick(sl, tick_size),
        trailing_callback_rate, activation_price
    )

    pos = {
        "symbol": symbol,
        "side": side,
        "entry_price": filled_price,
        "size": notional,
        "quantity": exec_qty,
        "signal_strength": strength,
        "rsi": features.get("rsi", 0.0),
        "ml_probability": probability,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "tp1_hit": False,
        "tp2_hit": False,
        "sl_synced": False,
        "trailing_active": False,
        "peak_price": filled_price,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "real",
        "order_type": "LIMIT",
        "status": "active"
    }
    for feature in REQUIRED_FEATURES:
        pos[feature] = features.get(feature, 0)

    positions.append(pos)
    save_positions(positions)

    approx_margin = notional / max(DEFAULT_LEVERAGE, 1)
    try:
        await send_telegram_message(
            f"🎯 <b>LIMIT POZİSYON AKTİF OLDU</b> {'🟢 LONG' if side=='long' else '🔴 SHORT'}\n"
            f"• Coin: <code>{tg_html(symbol)}</code>\n"
            f"• FILL Fiyatı: <code>{filled_price:.6f}</code>\n"
            f"• Miktar: <code>{exec_qty}</code> | Notional: <code>{notional:.2f} USDT</code> | Marj(~): <code>{approx_margin:.2f}</code>\n"
            f"• TP1/TP2/SL: <code>{tp1}</code> / <code>{tp2}</code> / <code>{sl}</code>\n"
            f"• ML Olasılık: <code>{probability*100:.1f}%</code>\n"
            f"• Zaman: <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>"
        )
    except Exception as e:
        logger.error(f"{symbol}: Limit fill bildirimi hatası: {e}")


# =========================
# Pozisyon Açma
# =========================
async def open_position(
    positions, symbol, side, price, rsi_val, ema_val, high, low, close, volume,
    strength, features, probability, notional_override=None
):
    if not ALLOW_MULTI_ENTRY_PER_SYMBOL:
        ex_qty, ex_side, _, _ = await get_current_position(symbol)
        if ex_qty and ex_qty > 0:
            logger.info(f"{symbol}: Borsada zaten açık pozisyon var (qty={ex_qty}). Yeni giriş engellendi.")
            return

        last_t = last_positions_time.get(symbol)
        if last_t:
            try:
                if (datetime.utcnow() - last_t).total_seconds() < REENTRY_COOLDOWN_MIN * 60:
                    logger.info(f"{symbol}: Re-entry cooldown aktif ({REENTRY_COOLDOWN_MIN}dk). Yeni giriş ertelendi.")
                    return
            except Exception:
                pass

    if any(p['symbol'] == symbol and not p.get('closed', False) for p in positions):
        return
    if len(positions) >= MAX_POSITIONS:
        return

    account_balance = await get_futures_balance()
    if account_balance < 5.0:
        logger.warning(f"Hesap bakiyesi düşük ({account_balance} USDT), pozisyon açılmadı.")
        return

    atr_val = float(talib.ATR(high, low, close, 14)[-1])

    if notional_override is not None:
        position_usdt = notional_override
    else:
        position_usdt = RiskCalculator.position_notional(account_balance, atr_val, float(price))
        
        # Regresyon tahminine göre boyutlandırma
        if USE_REGRESSION_FOR_SIZING and model_reg and "reg_prediction" in features:
            reg_pred = features.get("reg_prediction", 0)
            if reg_pred > 0:
                # Pozitif tahmin - boyutu artır (maksimum 2x)
                size_mult = min(1 + (reg_pred * 2), 2.0)
                position_usdt = position_usdt * size_mult
                logger.info(f"{symbol}: Regresyon bazlı pozisyon artırma: {reg_pred:.4f} → {size_mult:.2f}x")
            elif reg_pred < 0:
                # Negatif tahmin - boyutu azalt (minimum 0.5x)
                size_mult = max(1 - (abs(reg_pred) * 2), 0.5)
                position_usdt = position_usdt * size_mult
                logger.info(f"{symbol}: Regresyon bazlı pozisyon azaltma: {reg_pred:.4f} → {size_mult:.2f}x")

    precision, step_size, min_qty, ex_min_notional, tick_size = await get_symbol_trading_filters(symbol)
    env_min_notional = float(os.getenv("MIN_NOTIONAL_USDT", 5.0))
    target_min_notional = max(env_min_notional, ex_min_notional)

    quantity_raw = position_usdt / max(price, 1e-9)
    quantity = adjust_quantity(quantity_raw, precision, step_size)
    if quantity < max(step_size, min_qty):
        quantity = adjust_quantity_up(max(step_size, min_qty), precision, step_size)

    notional = quantity * price
    if notional + 1e-8 < target_min_notional:
        needed_qty_raw = target_min_notional / max(price, 1e-9)
        quantity_up = adjust_quantity_up(needed_qty_raw, precision, step_size)
        notional_up = quantity_up * price
        max_notional_allowed = account_balance * (MAX_NOTIONAL_PERCENT / 100.0)
        if notional_up <= max_notional_allowed:
            quantity = quantity_up
            notional = notional_up
        else:
            logger.warning(f"{symbol}: Min notional için gereken {notional_up:.2f} USDT, izin verilen üst sınırı aşıyor ({max_notional_allowed:.2f} USDT). Pozisyon açılmadı.")
            return

    await set_leverage(symbol, DEFAULT_LEVERAGE)

    if notional + 1e-8 < target_min_notional:
        logger.warning(f"{symbol}: Emir büyüklüğü {notional:.2f} USDT < minNotional {target_min_notional:.2f} USDT, açılmadı!")
        return

    # Limit emir kontrolü ve uygulaması
    use_limit_entry = env_bool("USE_LIMIT_ENTRY", False)
    entry_price = price
    order_type = "MARKET"
    
    if use_limit_entry:
        # Limit emir için fiyat ayarlaması (spread'i hesaba katarak)
        limit_offset_pct = float(os.getenv("LIMIT_OFFSET_PCT", "0.1"))
        if side == "long":
            limit_price = price * (1 - limit_offset_pct/100)
        else:
            limit_price = price * (1 + limit_offset_pct/100)
        
        limit_price = adjust_price_to_tick(limit_price, tick_size)
        
        # Limit emir gönder
        order = await open_binance_limit_position(symbol, side, quantity, limit_price)
        if not order:
            logger.warning(f"{symbol}: Limit emir açılamadı!")
            return

        order_id = order.get("orderId")
        # Pending kaydı (opsiyonel)
        if MARK_PENDING_AS_POSITION:
            pending_obj = {
                "symbol": symbol,
                "side": side,
                "entry_price": limit_price,
                "quantity": quantity,
                "size": quantity * limit_price,
                "status": "pending",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "order_id": order_id,
                "mode": "real",
                "order_type": "LIMIT_PENDING"
            }
            positions.append(pending_obj)
            save_positions(positions)

        # Fill bekle
        avg_price, exec_qty, orig_qty = await wait_for_limit_fill(symbol, order_id)
        if exec_qty <= 0:
            logger.info(f"{symbol}: Limit emir dolmadı, pozisyon açılmadı.")
            # Pending kaydı varsa sil
            positions[:] = [p for p in positions if not (p.get("symbol") == symbol and p.get("status") == "pending")]
            save_positions(positions)
            return

        filled_price = avg_price if avg_price else limit_price
        # Pending kaydı temizle
        positions[:] = [p for p in positions if not (p.get("symbol") == symbol and p.get("status") == "pending")]
        save_positions(positions)

        # TP/SL hesapla ve emirleri gönder
        tp1, tp2, sl = calc_tp_sl_abs(filled_price, atr_val, probability, side, tick_size)
        if not all([tp1, tp2, sl]):
            logger.warning(f"{symbol}: TP/SL hesaplanamadı!")
            return

        q_tp1_frac = env_float("TP1_FRACTION", 0.30)
        q_tp1 = adjust_quantity(exec_qty * q_tp1_frac, precision, step_size)
        q_tp2 = adjust_quantity(exec_qty - q_tp1, precision, step_size)
        q_sl = exec_qty

        trailing_callback_rate = max(0.1, min(5.0, TRAILING_OFFSET_ENV))
        activation_price = adjust_price_to_tick(tp2, tick_size)
        await open_tp_sl_orders(
            symbol, side, q_tp1, q_tp2, q_sl,
            adjust_price_to_tick(tp1, tick_size),
            adjust_price_to_tick(tp2, tick_size),
            adjust_price_to_tick(sl, tick_size),
            trailing_callback_rate, activation_price
        )

        # Pozisyon kaydını oluştur
        pos = {
            "symbol": symbol,
            "side": side,
            "entry_price": filled_price,
            "size": exec_qty * filled_price,
            "quantity": exec_qty,
            "signal_strength": strength,
            "rsi": rsi_val,
            "ml_probability": probability,
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "tp1_hit": False,
            "tp2_hit": False,
            "sl_synced": False,
            "trailing_active": False,
            "peak_price": filled_price,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "real",
            "order_type": "LIMIT"
        }
        for feature in REQUIRED_FEATURES:
            pos[feature] = features.get(feature, 0)

        positions.append(pos)
        save_positions(positions)

        last_positions_time[symbol] = datetime.utcnow()

        approx_margin = (exec_qty * filled_price) / max(DEFAULT_LEVERAGE, 1)
        entry_msg = (
            f"🎯 <b>YENİ POZİSYON</b> {'🟢 LONG' if side == 'long' else '🔴 SHORT'}\n"
            f"• Coin: <code>{tg_html(symbol)}</code>\n"
            f"• Emir Tipi: <code>LIMIT</code>\n"
            f"• Giriş: <code>{filled_price:.6f}</code>\n"
            f"• Boyut (Notional): <code>{exec_qty * filled_price:.2f} USDT</code> | Miktar: <code>{exec_qty}</code> | Marj(~): <code>{approx_margin:.2f} USDT</code>\n"
            f"• TP1/TP2/SL: <code>{tp1}</code> / <code>{tp2}</code> / <code>{sl}</code>\n"
            f"• ML Olasılık: <code>{probability*100:.1f}%</code>\n"
            f"• Zaman: <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>\n"
        )
        await send_telegram_message(entry_msg)
        return  # Limit akışı bitti

    else:
        # Normal market emirli açılış
        order = await open_binance_position(symbol, side, quantity)
        if order is None:
            logger.warning(f"{symbol}: Gerçek emir açılamadı!")
            return

        tp1, tp2, sl = calc_tp_sl_abs(entry_price, atr_val, probability, side, tick_size)
        if not all([tp1, tp2, sl]):
            logger.warning(f"{symbol}: TP/SL hesaplanamadı!")
            return

        q_tp1_frac = env_float("TP1_FRACTION", 0.30)
        q_tp1 = adjust_quantity(quantity * q_tp1_frac, precision, step_size)
        q_tp2 = adjust_quantity(quantity - q_tp1, precision, step_size)
        q_sl = quantity

        trailing_callback_rate = max(0.1, min(5.0, TRAILING_OFFSET_ENV))
        activation_price = adjust_price_to_tick(tp2, tick_size)
        await open_tp_sl_orders(
            symbol, side, q_tp1, q_tp2, q_sl,
            adjust_price_to_tick(tp1, tick_size),
            adjust_price_to_tick(tp2, tick_size),
            adjust_price_to_tick(sl, tick_size),
            trailing_callback_rate, activation_price
        )

        pos = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "size": notional,
            "quantity": quantity,
            "signal_strength": strength,
            "rsi": rsi_val,
            "ml_probability": probability,
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "tp1_hit": False,
            "tp2_hit": False,
            "sl_synced": False,
            "trailing_active": False,
            "peak_price": entry_price,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "real",
            "order_type": "MARKET"
        }
        for feature in REQUIRED_FEATURES:
            pos[feature] = features.get(feature, 0)

        positions.append(pos)
        save_positions(positions)

        last_positions_time[symbol] = datetime.utcnow()

        approx_margin = notional / max(DEFAULT_LEVERAGE, 1)
        entry_msg = (
            f"🎯 <b>YENİ POZİSYON</b> {'🟢 LONG' if side == 'long' else '🔴 SHORT'}\n"
            f"• Coin: <code>{tg_html(symbol)}</code>\n"
            f"• Emir Tipi: <code>MARKET</code>\n"
            f"• Giriş: <code>{entry_price:.6f}</code>\n"
            f"• Boyut (Notional): <code>{notional:.2f} USDT</code> | Miktar: <code>{quantity}</code> | Marj(~): <code>{approx_margin:.2f} USDT</code>\n"
            f"• TP1/TP2/SL: <code>{tp1}</code> / <code>{tp2}</code> / <code>{sl}</code>\n"
            f"• ML Olasılık: <code>{probability*100:.1f}%</code>\n"
            f"• Zaman: <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>\n"
        )
        await send_telegram_message(entry_msg)

async def check_spread_impact(symbol, entry_price):
    """Spread kontrolü - çok dar spreadli işlemleri engelle"""
    
    ticker = await binance_client.get_orderbook_ticker(symbol=symbol)
    bid = float(ticker['bidPrice'])
    ask = float(ticker['askPrice'])
    
    spread = ask - bid
    spread_percent = (spread / entry_price) * 100
    
    max_spread_percent = float(os.getenv("MAX_SPREAD_PERCENT", "0.02"))  # %0.02
    
    if spread_percent > max_spread_percent:
        logger.info(f"{symbol}: Spread çok geniş ({spread_percent:.3f}% > {max_spread_percent}%)")
        return False
    
    return True


# =========================
# Exit Emir Tamiri ve İzleme
# =========================
async def ensure_exit_orders_for_existing_position(pos) -> None:
    symbol = pos["symbol"]
    side = pos["side"]

    orders = await get_open_orders(symbol)
    ex_qty, ex_side, entry_px, mark_px = await get_current_position(symbol)
    if ex_side is None or ex_qty <= 0:
        return

    precision, step_size = await get_symbol_quantity_precision_and_step(symbol)
    _, _, _, _, tick_size = await get_symbol_trading_filters(symbol)

    qty_all = adjust_quantity(ex_qty, precision, step_size)

    q_tp1_frac = env_float("TP1_FRACTION", 0.30)
    q_tp1_frac = max(0.05, min(0.95, q_tp1_frac))
    qty_tp1 = adjust_quantity(qty_all * q_tp1_frac, precision, step_size)
    qty_tp2 = adjust_quantity(max(qty_all - qty_tp1, 0.0), precision, step_size)

    tp1 = adjust_price_to_tick(pos.get("tp1"), tick_size)
    tp2 = adjust_price_to_tick(pos.get("tp2"), tick_size)

    be_enabled = env_bool("TP1_BE_TO_ENTRY", True)
    be_offset_bp = env_float("TP1_BE_OFFSET_BP", 3.0)
    planned_sl = adjust_price_to_tick(pos.get("sl"), tick_size)
    if be_enabled and pos.get("tp1_hit"):
        entry_for_be = float(pos.get("entry_price") or entry_px or 0.0)
        if entry_for_be > 0:
            planned_sl = adjust_price_to_tick(be_price_from_entry(entry_for_be, side, be_offset_bp), tick_size)

    tp_orders = [o for o in orders if o.get("type") == "TAKE_PROFIT_MARKET"]
    has_tp = len(tp_orders) > 0
    sl_orders = [o for o in orders if o.get("type") == "STOP_MARKET"]
    has_sl = len(sl_orders) > 0
    trailing_orders = [o for o in orders if o.get("type") == "TRAILING_STOP_MARKET"]
    has_trailing = len(trailing_orders) > 0

    callback_cfg = max(0.1, min(5.0, TRAILING_OFFSET_ENV))
    trailing_ok = False
    if has_trailing:
        trailing_ok = _trailing_matches(
            trailing_orders=trailing_orders,
            qty_expected=qty_all,
            activation_expected=tp2,
            callback_expected=callback_cfg,
            tick_size=tick_size,
            step_size=step_size
        )

    try:
        min_interval = int(os.getenv("EXIT_SYNC_MIN_INTERVAL", "180"))
    except Exception:
        min_interval = 180
    now = datetime.utcnow()
    last_sync = None
    if pos.get("last_exit_sync_at"):
        try:
            last_sync = datetime.strptime(pos["last_exit_sync_at"], "%Y-%m-%d %H:%M:%S")
        except:
            pass
    if has_tp and has_sl and ((has_trailing and trailing_ok) or (not has_trailing and tp2 is None)):
        if last_sync and (now - last_sync).total_seconds() < min_interval:
            return

    client = await init_binance_client()
    if not client:
        return

    did_any = False
    try:
        close_side = "SELL" if side == "long" else "BUY"
        working_type = os.getenv("WORKING_PRICE_TYPE", "MARK_PRICE")
        use_price_protect = env_bool("PRICE_PROTECT", True)

        if not has_tp:
            if qty_tp1 > 0 and tp1:
                await client.futures_create_order(
                    symbol=symbol, side=close_side, type="TAKE_PROFIT_MARKET",
                    stopPrice=tp1, quantity=qty_tp1, reduceOnly=True,
                    workingType=working_type, priceProtect=use_price_protect
                ); did_any = True
            if qty_tp2 > 0 and tp2:
                await client.futures_create_order(
                    symbol=symbol, side=close_side, type="TAKE_PROFIT_MARKET",
                    stopPrice=tp2, quantity=qty_tp2, reduceOnly=True,
                    workingType=working_type, priceProtect=use_price_protect
                ); did_any = True

        if not has_sl and planned_sl and qty_all > 0:
            await client.futures_create_order(
                symbol=symbol, side=close_side, type="STOP_MARKET",
                stopPrice=planned_sl, quantity=qty_all, reduceOnly=True,
                workingType=working_type, priceProtect=use_price_protect
            ); did_any = True

        if not has_trailing and tp2:
            await client.futures_create_order(
                symbol=symbol, side=close_side, type="TRAILING_STOP_MARKET",
                quantity=qty_all, activationPrice=tp2, callbackRate=round(callback_cfg, 2),
                workingType=working_type, reduceOnly=True
            ); did_any = True
        elif has_trailing and not trailing_ok:
            for o in trailing_orders:
                try:
                    await client.futures_cancel_order(symbol=symbol, orderId=o.get("orderId"))
                except Exception as ce:
                    logger.warning(f"{symbol}: Trailing iptal edilemedi ({o.get('orderId')}): {ce}")
            await client.futures_create_order(
                symbol=symbol, side=close_side, type="TRAILING_STOP_MARKET",
                quantity=qty_all, activationPrice=tp2 if tp2 else None, callbackRate=round(callback_cfg, 2),
                workingType=working_type, reduceOnly=True
            ); did_any = True

        if did_any:
            pos["last_exit_sync_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"{symbol}: Exit emirleri onarıldı/senkronlandı (TP/SL/Trailing, BE={'ON' if (be_enabled and pos.get('tp1_hit')) else 'OFF'}).")
            await send_telegram_message(
                "🛠️ <b>Exit Emirleri Tamir/Senkron</b>\n"
                f"• Coin: <code>{tg_html(symbol)}</code>\n"
                f"• SL: <code>{planned_sl}</code> | TP1: <code>{tp1}</code> | TP2: <code>{tp2}</code>\n"
                f"• BE: <code>{'ON' if (be_enabled and pos.get('tp1_hit')) else 'OFF'}</code>\n"
                f"• Zaman: <code>{now.strftime('%H:%M:%S')} UTC</code>"
            )
        else:
            pos["last_exit_sync_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"{symbol}: Exit emir tamiri/senkron hatası: {e}")
    finally:
        try:
            await client.close_connection()
        except:
            pass

# =========================
# Pozisyon Takibi
# =========================
async def check_positions(positions):
    """
    Exchange odaklı izleme + eksik exit emirlerini tamir + ML kapatma kararları.
    """
    if not positions:
        return positions

    updated_positions = []
    for pos in positions:
        if pos.get('closed', False):
            continue
        if pos.get("status") == "pending":
            # Sadece zaman aşımı varsa temizle (örn 2 * LIMIT_FILL_TIMEOUT_SEC sonra)
            created = datetime.strptime(pos["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (datetime.utcnow() - created).total_seconds() > LIMIT_FILL_TIMEOUT_SEC + 5:
                # Çok uzun beklediyse sil (veya logla)
                logger.info(f"{pos['symbol']}: Pending limit fill timeout → kayıt siliniyor.")
                continue
            # Pending henüz pozisyona dönüşmedi, üzerinde işlem yapma
            updated_positions.append(pos)
            continue

        symbol = pos["symbol"]
        side = pos["side"]

        # 1) Eksik exit emirleri varsa tamamla
        try:
            await ensure_exit_orders_for_existing_position(pos)
        except Exception as e:
            logger.error(f"{symbol}: ensure_exit_orders hata: {e}")

        # 2) Borsadaki güncel pozisyon
        ex_qty, ex_side, entry_px, mark_px = await get_current_position(symbol)

        # 3) Anlık fiyat
        data = await fetch_klines(symbol, limit=1)
        current_price = data[-1]['close'] if data else (mark_px or pos.get("entry_price", 0.0))

        # 4) ML kapatma kararı kontrolü
        ml_prob = float(pos.get("current_ml_prob", pos.get("ml_probability", 1.0)))
        consec_low = int(pos.get("low_prob_count", 0))
        atr_pct = float(pos.get("atr_percent", 0.0))
        mark_px = float(current_price)

        do_close, reason = await should_ml_close(pos, ml_prob, consec_low, atr_pct, mark_px)
        if do_close:
            try:
                # Exit emirlerini iptal et
                await cancel_all_open_orders(symbol)

                # Güncel borsa miktarı ve yönünü tekrar al
                ex_qty, ex_side, entry_px2, mark_px2 = await get_current_position(symbol)

                ok = True
                if ex_side and ex_qty > 0:
                    ok = await close_position_market(symbol, ex_side, ex_qty)

                exit_px = current_price if current_price else (mark_px2 or entry_px or pos.get("entry_price", 0.0))

                if ok:
                    record_closed_trade(pos, exit_px, f"ML Kararıyla Kapatma ({reason})")
                    pos["closed"] = True
                    try:
                        await send_telegram_message(
                            "🤖 <b>ML Kararıyla Pozisyon Kapatıldı</b>\n"
                            f"• Coin: <code>{tg_html(symbol)}</code>\n"
                            f"• Olasılık: <code>{ml_prob*100:.1f}%</code>\n"
                            f"• Neden: <code>{reason}</code>\n"
                            f"• Kapanış Fiyatı: <code>{exit_px}</code>\n"
                            f"• Zaman: <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>"
                        )
                    except Exception as te:
                        logger.warning(f"{symbol}: ML close telegram send fail: {te}")
                    continue  # Bir sonraki pozisyona geç
                else:
                    logger.error(f"{symbol}: ML close MARKET emri başarısız oldu.")
            except Exception as e:
                logger.error(f"{symbol}: ML close işlemi hatası: {e}")

        # 5) Borsada pozisyon yoksa local kapat
        if ex_side is None or ex_qty <= 0:
            # Kapanış nedeni (TP/SL/Trailing) tahmini
            reason_hint = await infer_exchange_close_reason(symbol)
            reason_text = f" ({reason_hint})" if reason_hint else ""

            # Kapanış fiyatı
            exit_price = current_price if current_price else (mark_px or pos.get("entry_price", 0.0))

            # Yaklaşık PnL hesapla
            approx_pct, approx_usdt = compute_approx_pnl(pos, float(exit_price or 0.0))

            # Binance income üzerinden (varsa) gerçekleşen PnL
            try:
                open_time = datetime.strptime(pos.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
            except Exception:
                open_time = None
            realized_pnl = await get_realized_pnl_sum(symbol, start_dt=open_time)

            # Local history kaydı
            record_closed_trade(pos, exit_price, f"Exchange Exit{reason_text}")

            # Mesaj
            try:
                lines = [
                    f"✅ <b>Pozisyon Borsada Kapatıldı{reason_text}</b>",
                    f"• Coin: <code>{tg_html(symbol)}</code>",
                    f"• Giriş: <code>{tg_html(pos.get('entry_price'))}</code> | Kapanış: <code>{exit_price}</code>",
                    f"• Yaklaşık PnL: <code>{approx_pct:.2f}%</code> | ~<code>{approx_usdt} USDT</code>",
                ]
                if realized_pnl is not None:
                    lines.append(f"• Gerçekleşen (Binance): <code>{realized_pnl} USDT</code>")
                lines.append(f"• Zaman: <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>")
                await send_telegram_message("\n".join(lines))
            except Exception as te:
                logger.error(f"{symbol}: Kapanış bildirimi hatası: {te}")

            continue

        # 6) Yön uyuşmazlığı varsa güncelle
        if ex_side != side:
            logger.warning(f"{symbol}: Borsadaki yön ({ex_side}) local yön ({side}) ile uyuşmuyor. Local güncelleniyor.")
            pos["side"] = ex_side

        precision, step_size = await get_symbol_quantity_precision_and_step(symbol)
        try:
            _, _, _, _, tick_size = await get_symbol_trading_filters(symbol)
        except Exception:
            tick_size = None
        if not tick_size:
            tick_size = 0.0000001  

        # 7) TP1 detect ve SL/Trailing senkron + Break-Even
        orig_qty = float(pos.get("quantity") or 0.0)
        if orig_qty > 0 and not pos.get("tp1_hit"):
            q_tp1_frac = env_float("TP1_FRACTION", 0.30)
            q_tp1_frac = max(0.05, min(0.95, q_tp1_frac))

            # Ne kadarı kapanmış?
            try:
                executed_frac = (orig_qty - ex_qty) / orig_qty if orig_qty > 0 else 0.0
            except Exception:
                executed_frac = 0.0

            # Planlanan TP1 oranının ~%80'i kadar kapanma olduysa TP1 kabul et
            if executed_frac >= max(0.05, q_tp1_frac * 0.8):
                pos["tp1_hit"] = True
                pos["partial_tp1_done"] = True
                pos["tp1_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

                # Break-even ayarı
                be_enabled = env_bool("TP1_BE_TO_ENTRY", True)
                be_offset_bp = env_float("TP1_BE_OFFSET_BP", 3.0)
                entry_for_be = float(pos.get("entry_price") or entry_px or current_price)
                new_sl_price = pos.get("sl")

                if be_enabled and entry_for_be:
                    be_px = be_price_from_entry(entry_for_be, pos["side"], be_offset_bp)
                    new_sl_price = adjust_price_to_tick(be_px, tick_size)

                sl_sync_ok = False
                trailing_sync_ok = False
                try:
                    # SL'yi kalan miktar ve yeni fiyatla değiştir
                    sl_sync_ok = await replace_stop_loss_order(symbol, pos["side"], ex_qty, new_sl_price)
                    pos["sl"] = new_sl_price
                    pos["sl_synced"] = bool(sl_sync_ok)

                    # Trailing senkron: activation olarak TP2'yi kullan
                    trailing_sync_ok = await replace_trailing_order(symbol, pos["side"], ex_qty, pos.get("tp2"))
                    pos["trailing_active"] = True if trailing_sync_ok else pos.get("trailing_active", False)
                except Exception as e:
                    logger.error(f"{symbol}: SL/Trailing senkron hata: {e}")

                # Dinamik yüzdeli mesaj
                try:
                    closed_pct_txt = f"%{int(round(max(5.0, min(95.0, q_tp1_frac * 100.0))))}"
                    await send_telegram_message(
                        "✅ <b>TP1 Hedef</b> (" + closed_pct_txt + " kapandı)\n"
                        f"• Coin: <code>{tg_html(symbol)}</code>\n"
                        f"• Fiyat: <code>{current_price}</code>\n"
                        f"• Yeni SL: <code>{tg_html(new_sl_price)}</code> (BE{'+' if be_enabled else ''})\n"
                        f"• Kalan Miktar: <code>{ex_qty}</code>\n"
                        f"• SL Senkron: <code>{'OK' if sl_sync_ok else 'HATA'}</code>\n"
                        f"• Trailing Senkron: <code>{'OK' if trailing_sync_ok else 'HATA'}</code>\n"
                        f"• Zaman: <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>"
                    )
                except Exception as te:
                    logger.error(f"{symbol}: TP1 bildirimi hatası: {te}")

        # 8) TP2 eşiğinde trailing aktif işareti
        if not pos.get("tp2_hit"):
            hit_tp2 = (pos["side"] == "long" and current_price >= pos.get("tp2", float('inf'))) or \
                      (pos["side"] == "short" and current_price <= pos.get("tp2", -float('inf')))
            if hit_tp2:
                pos["tp2_hit"] = True
                pos["trailing_active"] = True
                pos["peak_price"] = current_price
                try:
                    await send_telegram_message(
                        "🎯 <b>TP2 Hedef</b> (Trailing SL aktif)\n"
                        f"• Coin: <code>{tg_html(symbol)}</code>\n"
                        f"• Fiyat: <code>{current_price}</code>\n"
                        f"• Zaman: <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>"
                    )
                except Exception as te:
                    logger.error(f"{symbol}: TP2 bildirimi hatası: {te}")

        # 9) Trailing izleme (bilgi amaçlı)
        if pos.get("trailing_active"):
            if pos["side"] == "long" and current_price > pos.get("peak_price", current_price):
                pos["peak_price"] = current_price
            elif pos["side"] == "short" and current_price < pos.get("peak_price", current_price):
                pos["peak_price"] = current_price

        # 10) Lokal miktar/notional'ı borsaya göre güncelle
        pos["quantity"] = ex_qty
        basis = entry_px if entry_px > 0 else current_price
        pos["size"] = round(ex_qty * basis, 6)

        updated_positions.append(pos)

    save_positions(updated_positions)
    return updated_positions

class OrderTracker:
    def __init__(self):
        self.tracked_orders = {}
        
    async def track_tp_sl_orders(self, symbol: str, pos: dict):
        """TP/SL emirlerini takip et ve bildir"""
        try:
            orders = await get_open_orders(symbol)
            
            tp_orders = [o for o in orders if o.get("type") == "TAKE_PROFIT_MARKET"]
            sl_orders = [o for o in orders if o.get("type") == "STOP_MARKET"]
            trailing_orders = [o for o in orders if o.get("type") == "TRAILING_STOP_MARKET"]
            
            # TP1 vurdu mu kontrol et
            if tp_orders and not pos.get("tp1_notified"):
                ex_qty, _, _, mark_px = await get_current_position(symbol)
                orig_qty = float(pos.get("original_quantity", pos.get("quantity", 0)))
                
                if orig_qty > 0 and ex_qty < orig_qty * 0.8:  # %20+ kapandıysa TP1 vurmuş
                    await notification_manager.send_tp_hit(symbol, "TP1", mark_px, ex_qty)
                    pos["tp1_notified"] = True
            
            # SL/Trailing durumunu bildir
            await self._notify_order_status(symbol, tp_orders, sl_orders, trailing_orders)
            
        except Exception as e:
            logger.error(f"Order tracking error for {symbol}: {e}")
    
    async def _notify_order_status(self, symbol, tp_orders, sl_orders, trailing_orders):
        """Emir durumu bildirimi"""
        status_msg = f"📊 <b>{symbol} EMİR DURUMU</b>\n"
        
        if tp_orders:
            status_msg += f"🎯 TP Emirleri: {len(tp_orders)} aktif\n"
        
        if sl_orders:
            status_msg += f"🛡️ SL Emirleri: {len(sl_orders)} aktif\n"
            
        if trailing_orders:
            for order in trailing_orders:
                callback = order.get("callbackRate", "N/A")
                activation = order.get("activationPrice", "N/A")
                status_msg += f"📈 Trailing: %{callback} (Aktivasyon: {activation})\n"
        
        # Sadece değişiklik varsa bildir
        current_hash = hash(status_msg)
        if self.tracked_orders.get(symbol) != current_hash:
            self.tracked_orders[symbol] = current_hash
            # await send_telegram_message(status_msg)  # İsteğe bağlı

order_tracker = OrderTracker()

class TelegramNotificationManager:
    def __init__(self):
        self.notification_queue = []
        self.last_sent = {}
        
    async def send_position_opened(self, symbol, side, entry_price, size, tp1, tp2, sl, ml_prob):
        """Pozisyon açılış bildirimi"""
        leverage_text = f"x{DEFAULT_LEVERAGE}" if DEFAULT_LEVERAGE > 1 else ""
        
        msg = (
            f"🎯 <b>YENİ POZİSYON AÇILDI</b> {'🟢 LONG' if side == 'long' else '🔴 SHORT'}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Coin: <code>{symbol}</code>\n"
            f"📈 Giriş: <code>{entry_price:.6f}</code> {leverage_text}\n"
            f"💵 Boyut: <code>{size:.2f} USDT</code>\n"
            f"🎯 TP1: <code>{tp1:.6f}</code> | TP2: <code>{tp2:.6f}</code>\n"
            f"🛡️ SL: <code>{sl:.6f}</code>\n"
            f"🤖 ML Güven: <code>{ml_prob*100:.1f}%</code>\n"
            f"⏰ <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>\n"
        )
        await self._send_with_retry(msg, "position_opened")
    
    async def send_tp_sl_orders_placed(self, symbol, tp1, tp2, sl, trailing_rate):
        """TP/SL emirleri yerleştirildi bildirimi"""
        msg = (
            f"📋 <b>EXIT EMİRLERİ YERLEŞTİRİLDİ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Coin: <code>{symbol}</code>\n"
            f"🎯 TP1: <code>{tp1:.6f}</code>\n"
            f"🎯 TP2: <code>{tp2:.6f}</code>\n"
            f"🛡️ SL: <code>{sl:.6f}</code>\n"
            f"📈 Trailing: <code>%{trailing_rate:.1f}</code>\n"
            f"⏰ <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>\n"
        )
        await self._send_with_retry(msg, "orders_placed")
    
    async def send_position_closed(self, symbol, side, entry_price, exit_price, pnl_pct, pnl_usdt, reason, commission_cost=None):
        """Pozisyon kapanış bildirimi"""
        profit_emoji = "✅" if pnl_usdt > 0 else "❌"
        
        msg = (
            f"{profit_emoji} <b>POZİSYON KAPANDI</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Coin: <code>{symbol}</code> ({'🟢 LONG' if side == 'long' else '🔴 SHORT'})\n"
            f"📊 Giriş: <code>{entry_price:.6f}</code>\n"
            f"📊 Çıkış: <code>{exit_price:.6f}</code>\n"
            f"💰 PnL: <code>{pnl_pct:+.2f}%</code> (<code>{pnl_usdt:+.4f} USDT</code>)\n"
        )
        
        if commission_cost:
            net_profit = pnl_usdt - commission_cost
            msg += f"💸 Komisyon: <code>-{commission_cost:.4f} USDT</code>\n"
            msg += f"💵 Net: <code>{net_profit:+.4f} USDT</code>\n"
        
        msg += (
            f"📝 Sebep: <code>{reason}</code>\n"
            f"⏰ <code>{datetime.utcnow().strftime('%H:%M:%S')} UTC</code>\n"
        )
        
        await self._send_with_retry(msg, "position_closed")
    
    async def _send_with_retry(self, message, msg_type, max_retries=3):
        """Retry mekanizması ile mesaj gönder"""
        for attempt in range(max_retries):
            try:
                await send_telegram_message(message)
                return True
            except Exception as e:
                logger.warning(f"Telegram send attempt {attempt+1} failed for {msg_type}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        logger.error(f"Failed to send {msg_type} notification after {max_retries} attempts")
        return False

# Global instance
notification_manager = TelegramNotificationManager()
adaptive_manager = None

async def replace_trailing_order(
    symbol: str,
    side: str,
    new_qty: float,
    activation_price: Optional[float],
    callback_rate: float = 0.5
) -> bool:
    """Var olan trailing stop emirlerini yeniden kurar."""
    try:
        if new_qty <= 0:
            return False
        client = await init_binance_client()
        if not client:
            return False
        try:
            info = await client.futures_exchange_info()
        except Exception as e:
            logger.error(f"[TRAILING] exchange_info hatası: {e}")
            return False
        symbols_info = {s["symbol"]: s for s in info["symbols"]}
        if symbol not in symbols_info:
            logger.error(f"[TRAILING] Sembol bulunamadı: {symbol}")
            return False
        filters = symbols_info[symbol]["filters"]
        price_tick = 0.0001
        qty_step = 0.001
        for f in filters:
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                price_tick = float(f["tickSize"])
            elif ft == "LOT_SIZE":
                qty_step = float(f["stepSize"])
        def round_step(val: float, step: float) -> float:
            if step <= 0: return val
            return math.floor(val / step) * step
        adj_qty = round_step(new_qty, qty_step)
        if adj_qty <= 0:
            logger.warning(f"[TRAILING] Ayarlanmış miktar 0: raw={new_qty} step={qty_step}")
            return False
        adj_activation = None
        if activation_price is not None:
            adj_activation = math.floor(activation_price / price_tick) * price_tick
        try:
            open_orders = await client.futures_get_open_orders(symbol=symbol)
        except Exception as oe:
            logger.error(f"[TRAILING] Açık emir sorgu hatası {symbol}: {oe}")
            return False
        cancel_count = 0
        for o in open_orders:
            if o.get("type") == "TRAILING_STOP_MARKET":
                try:
                    await client.futures_cancel_order(symbol=symbol, orderId=o.get("orderId"))
                    cancel_count += 1
                except Exception as ce:
                    logger.error(f"[TRAILING] İptal hatası {symbol} orderId={o.get('orderId')}: {ce}")
        logger.info(f"[TRAILING] {symbol} iptal edilen trailing emir sayısı: {cancel_count}")
        params = {
            "symbol": symbol,
            "side": side,
            "type": "TRAILING_STOP_MARKET",
            "quantity": adj_qty,
            "callbackRate": max(0.1, min(callback_rate, 5.0))
        }
        if adj_activation is not None:
            params["activationPrice"] = f"{adj_activation:.8f}"
        try:
            resp = await client.futures_create_order(**params)
            logger.info(f"[TRAILING] Yeni trailing emir {symbol} qty={adj_qty} activation={adj_activation} orderId={resp.get('orderId')}")
            return True
        except Exception as oe:
            logger.error(f"[TRAILING] Yeni trailing emir açma hatası {symbol}: {oe}", exc_info=True)
            return False
    except Exception as e:
        logger.error(f"[TRAILING] replace_trailing_order genel hata: {e}", exc_info=True)
        return False

async def calculate_open_pnl(positions):
    if not positions:
        return None, None, []
    total_pnl = 0.0
    pnl_details = []
    position_counts = {"long": 0, "short": 0}
    for pos in positions:
        data = await fetch_klines(pos["symbol"], limit=1)
        if not data:
            continue
        current_price = data[-1]['close']
        entry = pos["entry_price"]
        side = pos["side"]
        position_counts[side] += 1
        pnl = (current_price - entry)/entry*100 if side == "long" else (entry - current_price)/entry*100
        total_pnl += pnl * (pos["size"]/10)
        pnl_details.append({
            "symbol": pos["symbol"],
            "side": side,
            "pnl": round(pnl, 2),
            "size": pos["size"]
        })
    avg_pnl = total_pnl / len(positions) if positions else 0.0
    return avg_pnl, position_counts, sorted(pnl_details, key=lambda x: abs(x["pnl"]), reverse=True)

# =========================
# Yönetim ve Raporlama
# =========================
def build_open_pnl_report_html(pnl_details, avg_pnl, position_counts, ts_utc=None):
    if ts_utc is None:
        ts_utc = datetime.utcnow().strftime('%H:%M:%S')
    lines = ["📊 <b>Açık Pozisyon PnL Raporu</b>", "-----------------------------"]
    for d in pnl_details:
        sym = tg_html(d.get("symbol", "-"))
        side = tg_html((d.get("side", "") or "").upper())
        pnl = d.get("pnl", 0.0)
        lines.append(f"• <code>{sym}</code> (<code>{side}</code>): <code>{pnl:.2f}%</code>")
    lines.extend([
        "-----------------------------",
        f"💰 <b>Toplam Anlık PnL</b>: <code>{avg_pnl:.2f}%</code>",
        f"📊 Dağılım: <code>{position_counts.get('long',0)}L</code> | <code>{position_counts.get('short',0)}S</code>",
        f"🕒 <code>{ts_utc} UTC</code>"
    ])
    return "\n".join(lines)

async def manage_positions():
    if model_cls is None:
        return

    positions = load_positions()
    if not positions:
        return

    now = datetime.utcnow()
    # Açılış sonrası "grace"
    if (now - STARTUP_AT).total_seconds() < STARTUP_GRACE_MINUTES * 60:
        return

    symbols = [p["symbol"] for p in positions if not p.get('closed', False)]
    if not symbols:
        return

    df = await build_features_dataframe(symbols)
    if df.empty:
        return

    # Özellik ve olasılık haritası
    feature_map = df.set_index("symbol").to_dict(orient="index")
    
    # Güncellenmiş özellik seçimi ve ML shape kontrolü
    X = select_features_frame(df)
    n_expected = getattr(model_cls, "n_features_in_", len(REQUIRED_FEATURES))
    if X.shape[1] != n_expected:
        logger.warning(f"ML shape guard (manage_positions): X.shape[1]={X.shape[1]} != expected={n_expected}. REQUIRED_FEATURES={REQUIRED_FEATURES}")
        X = X.iloc[:, :n_expected]

    # Olasılık hesaplama ve inversion ayarı
    probs = model_cls.predict_proba(X)[:, 1]
    if is_invert_prob():
        probs = 1.0 - probs
    probs_by_symbol = {sym: float(p) for sym, p in zip(df["symbol"].tolist(), probs)}

    kept_positions = []
    any_update = False

    for pos in positions:
        if pos.get('closed', False):
            continue

        sym = pos["symbol"]

        # Özellikleri pozisyona işle (özellikle atr_percent)
        feats = feature_map.get(sym)
        if feats:
            try:
                pos["atr_percent"] = float(feats.get("atr_percent", pos.get("atr_percent", 0.0)))
            except Exception:
                pass

        prob = probs_by_symbol.get(sym)
        if prob is None:
            kept_positions.append(pos)
            continue

        # Olasılık ve sayaç
        pos["current_ml_prob"] = prob
        low = prob < ML_CLOSE_THRESHOLD
        cnt = int(pos.get("low_prob_count", 0))
        cnt = cnt + 1 if low else 0
        pos["low_prob_count"] = cnt

        # Pozisyon açılışına "open grace"
        try:
            open_time = datetime.strptime(pos["timestamp"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            open_time = None
        if open_time and (now - open_time).total_seconds() < OPEN_GRACE_MINUTES * 60:
            kept_positions.append(pos)
            continue

        # Fiyat ve PnL
        data = await fetch_klines(sym, limit=1)
        current_price = data[-1]['close'] if data else pos.get("entry_price", 0.0)
        side = pos["side"]
        entry = pos.get("entry_price", 0.0)
        if entry and entry > 0:
            pnl_pct = ((current_price - entry) / entry * 100.0) if side == "long" else ((entry - current_price) / entry * 100.0)
        else:
            pnl_pct = 0.0

        # Opsiyonel negatif PnL şartı
        if ML_CLOSE_REQUIRE_NEG_PNL and pnl_pct >= 0:
            kept_positions.append(pos)
            continue

        # should_ml_close ile nihai karar (ATR guard, BE koruması vb. dahildir)
        atr_pct = float(pos.get("atr_percent", 0.0))
        do_close, reason = await should_ml_close(pos, prob, cnt, atr_pct, float(current_price))

        if not do_close or cnt < ML_CLOSE_MIN_CONSECUTIVE:
            kept_positions.append(pos)
            continue

        # Borsadaki güncel miktarı al (TP1 sonrası değişmiş olabilir)
        ex_qty, ex_side, entry_px, mark_px = await get_current_position(sym)
        if ex_side is None or ex_qty <= 0:
            # Borsada pozisyon yoksa local kapat
            exit_price = current_price if current_price else (mark_px or entry)
            record_closed_trade(pos, exit_price, "ML Tahminiyle Kapatma (borsada pozisyon yok)")
            pos["closed"] = True
            any_update = True
            continue

        # Kapatma bildirimi
        reason_text = pos.get("force_close_reason") or f"ml_close:{reason}"
        try:
            await send_telegram_message(
                "⚠️ <b>Pozisyon Kapatma</b>\n"
                f"• Coin: <code>{tg_html(sym)}</code>\n"
                f"• Neden: <code>{reason_text}</code>"
            )
        except Exception as te:
            logger.warning(f"{sym}: force-close telegram gönderilemedi: {te}")

        # Exit emirlerini iptal et ve kapat
        await cancel_all_open_orders(sym)
        ok = await close_position_market(sym, ex_side, ex_qty)

        exit_price = current_price if current_price else (mark_px or entry)
        if ok:
            record_closed_trade(pos, exit_price, "ML Tahminiyle Kapatma")
            pos["closed"] = True
            any_update = True
            try:
                await send_telegram_message(
                    "🤖 <b>ML Kararıyla Pozisyon Kapatıldı</b>\n"
                    f"• Coin: <code>{_escape(sym, quote=False)}</code>\n"
                    f"• Olasılık: <code>{prob*100:.1f}%</code>\n"
                    f"• Kapanış Fiyatı: <code>{exit_price}</code>\n"
                    f"• Zaman: <code>{now.strftime('%H:%M:%S')} UTC</code>"
                )
            except Exception as te:
                logger.error(f"ML kapatma Telegram hatası: {te}")
        else:
            logger.error(f"{sym}: ML kapanış MARKET emri başarısız oldu.")
            kept_positions.append(pos)

    kept_positions = [p for p in kept_positions if not p.get('closed', False)]
    save_positions(kept_positions)

def SymbolRanker_rank_symbols(symbols, historical_data):
    ranked = []
    for sym in symbols:
        data = historical_data.get(sym)
        if not data:
            continue
        vol = data.get('volume_24h', 0.0)
        atr = data.get('atr', 0.0)
        liquidity_ok = vol > max(10000, LIQUIDITY_THRESHOLD * 0.3)
        if liquidity_ok:
            score = (atr * 0.4) + (np.log(max(vol, 1)) * 0.6)
            
            # BTC ve ETH'i biraz daha alt sıralara it
            if sym in ["BTCUSDT", "ETHUSDT"]:
                score *= 0.8  # Skoru %20 düşür
            
            # Çeşitlendirme için rastgele bir faktör ekle (%10 etki)
            score *= (0.95 + random.random() * 0.1)
            
            ranked.append((sym, score))
    return sorted(ranked, key=lambda x: x[1], reverse=True)

async def fetch_historical_data(symbols):
    historical_data = {}
    for symbol in symbols:
        try:
            klines = await fetch_klines(symbol, limit=300)
            if len(klines) < 50:
                continue
            closes = np.array([k['close'] for k in klines], dtype=np.float64)
            highs = np.array([k['high'] for k in klines], dtype=np.float64)
            lows = np.array([k['low'] for k in klines], dtype=np.float64)
            volumes = np.array([k['volume'] for k in klines], dtype=np.float64)
            historical_data[symbol] = {
                'atr': float(talib.ATR(highs, lows, closes, 14)[-1]),
                'volume_24h': float(np.sum(volumes[-24*12:])),
                'rsi': float(talib.RSI(closes, 14)[-1])
            }
        except Exception as e:
            logger.error(f"{symbol} veri çekme hatası: {str(e)}")
            continue
    return historical_data

# =========================
# Looplar
# =========================
async def monitor_positions_loop():
    global _last_adapt_eval
    last_report_time = datetime.utcnow()
    while True:
        try:
            positions = load_positions()
            positions = await check_positions(positions)

            # ML yönetimini her döngüde çalıştır
            if positions:
                await manage_positions()

                now = datetime.utcnow()
                # PnL raporunu belirli aralıkla gönder
                if (now - last_report_time).total_seconds() >= PNL_REPORT_INTERVAL:
                    avg_pnl, position_counts, pnl_details = await calculate_open_pnl(positions)
                    if avg_pnl is not None:
                        msg = build_open_pnl_report_html(
                            pnl_details=pnl_details,
                            avg_pnl=round(avg_pnl, 2),
                            position_counts=position_counts,
                            ts_utc=now.strftime('%H:%M:%S')
                        )
                        await send_telegram_message(msg)
                        last_report_time = now

            # Adaptif değerlendirme - her 15 dakikada bir
            if adaptive_manager and adaptive_manager.enabled:
                now = datetime.utcnow()
                if _last_adapt_eval is None:
                    _last_adapt_eval = now  # İlk turda başlat
                elapsed = (now - _last_adapt_eval).total_seconds()
                if elapsed >= ADAPTIVE_EVAL_INTERVAL:
                    try:
                        start_t = now
                        adaptive_manager.evaluate_and_select()
                        _last_adapt_eval = datetime.utcnow()
                        logger.debug(f"[ADAPT-PARAM] evaluate_and_select tamam (elapsed={int(elapsed)}s, duration={(datetime.utcnow()-start_t).total_seconds():.3f}s)")
                    except Exception as e:
                        logger.error(f"[ADAPT-PARAM] evaluate hata: {e}", exc_info=True)

            await asyncio.sleep(60)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"monitor_positions_loop hata: {e}", exc_info=True)
            await asyncio.sleep(5)

def generate_pnl_report(days=1):
    if not os.path.exists(HISTORY_FILE):
        return "📉 <b>Hiç kapanmış işlem bulunamadı.</b>"
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days)

    # Sadece 'real' (veya training) kapanışlar: shadow verileri hariç
    filtered = [
        t for t in history
        if datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff
        and (t.get("mode") in (None, "real", "training"))
        and t.get("closed_reason") != "shadow"
    ]
    if not filtered:
        return f"📉 <b>Son {days} gün içinde işlem yok.</b>"

    total_pnl = sum(float(t.get("profit_usdt", 0.0) or 0.0) for t in filtered)
    win_count = sum(1 for t in filtered if float(t.get("profit_usdt", 0.0) or 0.0) > 0)
    loss_count = sum(1 for t in filtered if float(t.get("profit_usdt", 0.0) or 0.0) < 0)
    label = "Günlük" if days == 1 else "Haftalık" if days == 7 else "Aylık" if days == 30 else f"Son {days} Günlük"
    report = (
        f"📊 <b>{label} İşlem Özeti</b>\n"
        f"-----------------------------\n"
        f"✅ Kar Eden İşlem: <code>{win_count}</code>\n"
        f"❌ Zarar Eden İşlem: <code>{loss_count}</code>\n"
        f"💰 Net PnL: <code>{total_pnl:.2f} USDT</code>\n"
        f"📈 Toplam İşlem: <code>{len(filtered)}</code>\n"
        f"🕒 Rapor Zamanı: <code>{now.strftime('%Y-%m-%d %H:%M:%S')} UTC</code>\n"
    )
    return report

def _parse_daily_report_time(s: str):
    try:
        hh, mm = s.strip().split(":")
        return int(hh), int(mm)
    except Exception:
        return 0, 0  # fallback midnight

async def periodic_pnl_report_loop():
    sent_today = None
    while True:
        now_local = datetime.utcnow() + timedelta(hours=3)  # Örn. TR saati gibi
        hh, mm = _parse_daily_report_time(DAILY_REPORT_TIME)
        if now_local.hour == hh and now_local.minute == mm:
            if sent_today != now_local.date():
                daily = generate_pnl_report(days=1)
                await send_telegram_message(daily)
                sent_today = now_local.date()
                await asyncio.sleep(60)
        await asyncio.sleep(10)

async def weekly_monthly_report_loop():
    last_sent_week = None
    last_sent_month = None
    while True:
        now = datetime.utcnow()
        if now.weekday() == 0 and now.hour == 10:
            if last_sent_week != now.date():
                weekly_report = generate_pnl_report(days=7)
                await send_telegram_message("📊 <b>Haftalık Kâr/Zarar Raporu</b>\n" + weekly_report)
                last_sent_week = now.date()
            if now.day <= 7 and last_sent_month != now.month:
                monthly_report = generate_pnl_report(days=30)
                await send_telegram_message("📅 <b>Aylık Kâr/Zarar Raporu</b>\n" + monthly_report)
                last_sent_month = now.month
            await asyncio.sleep(3600)
        await asyncio.sleep(60)

async def enhanced_trading_loop():
    global last_scanned, risk_manager, market_analyzer
    
    # Risk manager ve market analyzer'ı başlat
    if risk_manager is None:
        risk_manager = RiskManager(
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", 2.0)),
            max_weekly_loss_pct=float(os.getenv("MAX_WEEKLY_LOSS_PCT", 5.0))
        )
    
    if market_analyzer is None:
        market_analyzer = MarketStateAnalyzer()
    
    while True:
        try:
            # Güncel piyasa durumunu analiz et
            client = await init_binance_client()
            if client:
                market_state, volatility = await market_analyzer.analyze_market(client)
                await client.close_connection()
            else:
                market_state, volatility = "unknown", "normal"
            
            # Güncel hesap bakiyesini al
            account_balance = await get_futures_balance()
            
            # Risk kontrolü - maksimum kayıp sınırına ulaşıldı mı?
            can_trade, reason = risk_manager.should_trade(account_balance)
            if not can_trade:
                logger.warning(f"Trading durduruldu: {reason}")
                await send_telegram_message(f"⚠️ <b>Trading Durduruldu</b>\n• Sebep: {reason}")
                await asyncio.sleep(3600)  # 1 saat bekle
                continue
            
            # Piyasa durumuna göre parametre ayarlamaları
            position_params = market_analyzer.get_position_adjustment()
            strategy_params = market_analyzer.get_strategy_adjustment()
            
            effective_max_positions = int(MAX_POSITIONS * position_params["size_multiplier"])
            
            # Mevcut pozisyonları yönet
            positions = load_positions()
            active_positions = [p for p in positions if not p.get("closed", False)]
            
            # Pozisyon sayısı limitte mi?
            if len(active_positions) >= effective_max_positions:
                await asyncio.sleep(300)  # 5 dakika bekle
                continue
            
            # Sembol listesini al ve sırala
            symbols = await fetch_symbols()
            logger.debug(f"Trading loop: {len(symbols)} sembol taranacak")
            liquid_symbols = []
            for s in symbols:
                data = await fetch_liquidity_data(s)
                if data.get('volume_24h', 0) > LIQUIDITY_THRESHOLD:
                    liquid_symbols.append(s)
            logger.debug(f"Trading loop: Likidite filtresi sonrası {len(liquid_symbols)} sembol kaldı")
            logger.debug(f"İlk 10 likit sembol: {liquid_symbols[:10]}")
            
            # Tarihsel veri ve sıralama
            historical_data = await fetch_historical_data(liquid_symbols)
            ranked_symbols = SymbolRanker_rank_symbols(liquid_symbols, historical_data)
            
            # Piyasa durumuna göre sınırlı sembol listesi
            ranked_symbols = ranked_symbols[:100]
            
            # Herbir sembolü kontrol et
            for symbol, score in ranked_symbols:
                # Limiti aştık mı?
                if len(active_positions) >= effective_max_positions:
                    break
                
                # Mevcut pozisyon kontrolü
                if not ALLOW_MULTI_ENTRY_PER_SYMBOL and any(p['symbol'] == symbol for p in active_positions):
                    continue
                
                # Son taramada var mı?
                if symbol in last_scanned[-5:]:
                    await asyncio.sleep(1)
                    continue
                    
                last_scanned.append(symbol)
                if len(last_scanned) > 10:
                    last_scanned = last_scanned[-10:]
                
                # Teknik analiz verileri
                klines = await fetch_klines(symbol)
                if not klines or len(klines) < 50:
                    continue
                    
                close = np.array([k['close'] for k in klines], dtype=np.float64)
                high = np.array([k['high'] for k in klines], dtype=np.float64)
                low = np.array([k['low'] for k in klines], dtype=np.float64)
                open_ = np.array([k['open'] for k in klines], dtype=np.float64)
                volume = np.array([k['volume'] for k in klines], dtype=np.float64)
                
                # Teknik göstergeler
                try:
                    rsi = talib.RSI(close, 14)
                    ema = talib.EMA(close, 20)
                    macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                    upper_bb, middle_bb, lower_bb = talib.BBANDS(close, timeperiod=20)
                    atr_arr = talib.ATR(high, low, close, 14)
                except Exception:
                    continue
                
                # Piyasa durumuna göre RSI parametreleri
                rsi_low = strategy_params["rsi_low"]
                rsi_high = strategy_params["rsi_high"]
                
                # Sinyal kontrolü (adaptif parametrelerle)
                side, strength, features, probability = await check_signal(
                    symbol, rsi, ema, close, volume, open_, high, low, 
                    macd, macd_signal, upper_bb, lower_bb
                )
                
                if side:
                    # Pozisyon boyutunu piyasa durumuna göre ayarla
                    position_size_multiplier = position_params["size_multiplier"]
                    
                    # Yön bazlı kontrol
                    position_counts = await count_open_by_side_exchange()
                    if side == "long" and position_counts["long"] >= MAX_LONG_POSITIONS:
                        logger.info(f"{symbol}: Long pozisyon limiti doldu ({position_counts['long']}/{MAX_LONG_POSITIONS})")
                        continue
                    if side == "short" and position_counts["short"] >= MAX_SHORT_POSITIONS:
                        logger.info(f"{symbol}: Short pozisyon limiti doldu ({position_counts['short']}/{MAX_SHORT_POSITIONS})")
                        continue
                    
                    # Pozisyon açma işlemi (volatilite ayarlı risk)
                    account_balance = await get_futures_balance()
                    atr_val = float(atr_arr[-1])
                    
                    # Pozisyon boyutlandırma
                    position_usdt = RiskCalculator.position_notional(
                        account_balance, 
                        atr_val, 
                        float(close[-1])
                    ) * position_size_multiplier
                    
                    # Pozisyonu aç
                    await open_position(
                        positions, symbol, side, float(close[-1]), float(rsi[-1]), float(ema[-1]),
                        high, low, close, volume, strength, features, probability,
                        notional_override=position_usdt
                    )
                    
                    # İşlemi risk yöneticisine kaydet
                    risk_manager.record_trade(0)  # Başlangıçta 0, kapanışta güncellenecek

                    processed_symbols = {}
                    rejected_reasons = {}
                    await asyncio.sleep(1)
            
            # Döngü arasında bekleme
            adjusted_cooldown = COOLDOWN_MINUTES * 60
            # Yüksek volatilitede daha sık tara
            if volatility == "high":
                adjusted_cooldown = adjusted_cooldown * 0.6
            await asyncio.sleep(adjusted_cooldown)
            
        except Exception as e:
            logger.exception("Enhanced trading loop error")
            await asyncio.sleep(60)

def _tg(s) -> str:
    # Telegram HTML parse_mode için güvenli kaçış
    return _escape(str(s), quote=False)

def build_start_message():
    try:
        models_status = "Aktif" if model_cls else "Pasif"
    except NameError:
        models_status = "Bilinmiyor"
    
    # Adaptif yönetim kontrolü
    adapt_line = ""
    if 'adaptive_manager' in globals() and adaptive_manager and adaptive_manager.enabled:
        ap = adaptive_manager.get_runtime_params()
        adapt_line = (f"• Adaptif RSI: {ap.get('profile_rsi')} ({ap.get('rsi_low')}-{ap.get('rsi_high')}) | "
                      f"TPProf: {ap.get('profile_tp')} (TP1x={ap.get('atr_tp1_multiplier')}, SLx={ap.get('atr_sl_multiplier')})\n")

    return (
        f"🤖 <b>BOT AKTİF</b> (V5.1 - Futures, Stabil TP/SL, ML Fix)\n"
        f"• Başlangıç Zamanı: <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</code>\n"
        f"• Maks. Pozisyon: <code>{_tg(MAX_POSITIONS)}</code> | Cooldown: <code>{_tg(COOLDOWN_MINUTES)}dk</code>\n"
        f"• ML Threshold: <code>{_tg(ML_THRESHOLD)}</code> | Training Boyutu: <code>{_tg(TRAINING_POSITION_SIZE)} USDT</code>\n"
        f"• ML Modeller: <code>{_tg(models_status)}</code>\n"
        + adapt_line
    )

def build_existing_positions_message(positions: list) -> str:
    lines = ["🔍 <b>Mevcut Pozisyonlar Tespit Edildi:</b>"]
    any_listed = False
    for pos in positions:
        if pos.get("closed", False):
            continue
        qty = float(pos.get("quantity") or 0.0)
        if qty <= 0:
            continue

        side = (pos.get("side") or "").lower()
        side_emoji = "🟢" if side == "long" else "🔴"
        mode_emoji = "🧪" if pos.get("mode") == "training" else "🎯"

        symbol = tg_html(pos.get("symbol", "-"))
        entry = tg_html(pos.get("entry_price", "-"))
        tp1 = tg_html(pos.get("tp1", "-"))
        tp2 = tg_html(pos.get("tp2", "-"))
        sl = tg_html(pos.get("sl", "-"))
        side_up = tg_html(side.upper() if side else "-")

        lines.append(
            f"{mode_emoji} {side_emoji} <code>{symbol}</code> ({side_up}):\n"
            f"• Entry: <code>{entry}</code>\n"
            f"• TP1: <code>{tp1}</code> | TP2: <code>{tp2}</code>\n"
            f"• SL: <code>{sl}</code>\n"
        )
        any_listed = True

    if not any_listed:
        lines.append("• (Açık pozisyon yok)")
    return "\n".join(lines)

# =========================
# Senkronizasyon
# =========================
async def sync_positions_from_binance():
    client = await init_binance_client()
    if not client:
        logger.warning("sync_positions_from_binance: Binance client yok, senkron atlandı.")
        return

    try:
        account = await client.futures_account()
        open_positions = []
        for p in account.get("positions", []):
            try:
                symbol = p.get("symbol")
                if not symbol or not symbol.endswith("USDT"):
                    continue
                amt = float(p.get("positionAmt", 0.0) or 0.0)
                if abs(amt) < 1e-9:
                    continue

                side = "long" if amt > 0 else "short"
                entry_price = float(p.get("entryPrice", 0.0) or 0.0)
                mark_price = float(p.get("markPrice", 0.0) or 0.0)
                qty = abs(amt)
                basis_price = entry_price if entry_price > 0 else mark_price
                notional = qty * basis_price

                try:
                    orders = await client.futures_get_open_orders(symbol=symbol)
                except Exception:
                    orders = []

                tp_candidates = []
                sl_price = None
                trailing_active = False
                for o in orders:
                    typ = o.get("type")
                    sp = o.get("stopPrice") or o.get("stop_price") or o.get("price")
                    if typ == "TAKE_PROFIT_MARKET" and sp is not None:
                        tp_candidates.append(float(sp))
                    elif typ == "STOP_MARKET" and sp is not None:
                        sl_price = float(sp)
                    elif typ == "TRAILING_STOP_MARKET":
                        trailing_active = True

                tp1 = tp2 = None
                if tp_candidates:
                    if side == "long":
                        tp1 = min(tp_candidates)
                        tp2 = max(tp_candidates)
                    else:
                        tp1 = max(tp_candidates)
                        tp2 = min(tp_candidates)

                pos_obj = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": basis_price,
                    "size": round(notional, 6),
                    "quantity": qty,
                    "signal_strength": 0,
                    "rsi": 0.0,
                    "ml_probability": 0.0,
                    "tp1": tp1,
                    "tp2": tp2,
                    "sl": sl_price,
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "sl_synced": False,
                    "trailing_active": trailing_active,
                    "peak_price": basis_price,
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "real",
                }
                for feature in REQUIRED_FEATURES:
                    pos_obj.setdefault(feature, 0)
                open_positions.append(pos_obj)
            except Exception as e:
                logger.error(f"sync_positions_from_binance: pozisyon parse hata ({p.get('symbol')}): {e}")

        local = load_positions()
        local_symbols = {p["symbol"] for p in local if not p.get("closed", False)}
        merged = local[:]
        new_added = 0
        for pos in open_positions:
            if pos["symbol"] not in local_symbols:
                merged.append(pos)
                new_added += 1

        if new_added > 0:
            save_positions(merged)
            logger.info(f"🔄 Binance'ten {new_added} açık pozisyon senkronize edildi.")
            try:
                lines = ["🔄 <b>Binance Senkronizasyonu</b>: Açık pozisyonlar eklendi."]
                for pos in open_positions:
                    lines.append(
                        f"• <code>{tg_html(pos['symbol'])}</code> ({pos['side'].upper()}) @ <code>{pos['entry_price']}</code>"
                    )
                await send_telegram_message("\n".join(lines))
            except Exception as e:
                logger.error(f"Senkron bildirim hatası: {e}")
        else:
            logger.info("🔄 Binance senkron: Eklenebilecek yeni pozisyon yok.")
    except Exception as e:
        logger.error(f"sync_positions_from_binance hata: {e}", exc_info=True)
    finally:
        await client.close_connection()

async def debug_symbol_list():
    """Mevcut sembol listesini ve filtrelenmiş halini göster"""
    symbols = await fetch_symbols()
    logger.debug(f"TÜM SEMBOLLER ({len(symbols)}): {symbols[:10]}...")
    
    client = await init_binance_client()
    if client:
        historical_data = {}
        for symbol in symbols[:20]:  # İlk 20 sembolü kontrol et
            data = await fetch_liquidity_data(symbol)
            historical_data[symbol] = data
            logger.debug(f"{symbol}: hacim={data.get('volume_24h', 0):.2f} USDT")
        await client.close_connection()

async def main():
    global STARTUP_AT, risk_manager, market_analyzer, adaptive_manager
    STARTUP_AT = datetime.utcnow()
    initialize_files()
    load_models()
    adaptive_manager = AdaptiveParameterManager(history_file=HISTORY_FILE, logger=logger)
    config = load_config()
    risk_manager = RiskManager(
        max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", config["risk"]["max_daily_loss_percent"])),
        max_weekly_loss_pct=float(os.getenv("MAX_WEEKLY_LOSS_PCT", config["risk"]["max_weekly_loss_percent"]))
    )
    market_analyzer = MarketStateAnalyzer()
    print("\033[92m" + "="*50)
    print("🤖 BOT AKTİF (Stabil Sürüm)")
    print(f"• Başlangıç Zamanı: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"• Maks. Pozisyon: {MAX_POSITIONS} | Cooldown: {COOLDOWN_MINUTES}dk")
    print(f"• ML Threshold: {os.getenv('ML_THRESHOLD', ML_THRESHOLD)} | Training Boyutu: {TRAINING_POSITION_SIZE} USDT")
    print("="*50 + "\033[0m")
    try:
        await send_telegram_message(build_start_message())
    except Exception as e:
        logger.error(f"Başlangıç mesajı gönderilemedi: {e}")
    try:
        await sync_positions_from_binance()
        await prune_local_positions_not_on_exchange(send_notice=True)
    except Exception as e:
        logger.error(f"Senkron/Temizlik hatası: {e}")
    try:
        positions = load_positions()
        if positions:
            await send_telegram_message(build_existing_positions_message(positions))
    except Exception as e:
        logger.error(f"Pozisyon bildirimi hatası: {e}")
    try:
        tasks = [
            enhanced_trading_loop(),
            monitor_positions_loop(),
            periodic_pnl_report_loop(),
            weekly_monthly_report_loop(),
        ]
        if env_bool("ADAPTIVE_ML_THRESHOLD_ENABLE", False):
            tasks.append(adaptive_ml_threshold_loop())
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"main: Fatal hata: {e}")
        await asyncio.sleep(30)
        await main()

if __name__ == "__main__":
    asyncio.run(main())
