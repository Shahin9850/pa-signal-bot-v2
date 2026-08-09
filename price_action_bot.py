"""
ربات سیگنال‌دهی بر پایه پرایس‌اکشن خالص (بدون اندیکاتور)
شامل: ساختار بازار (BOS/CHoCH)، الگوهای کندلی، مناطق عرضه/تقاضا،
Order Block، Liquidity Grab
+ محاسبه دقیق حجم معامله، اهرم و ریسک بر اساس سرمایه مشخص‌شده
+ یادگیری خودکار: ربات نتیجه سیگنال‌های قبلی را با چک‌کردن کندل‌های
  بعدی می‌سنجد و وزن هر فاکتور تحلیلی را بر اساس نرخ موفقیت واقعی‌اش
  به‌مرور تنظیم می‌کند (نه بعد از یک اشتباه، بلکه بعد از تجمیع آمار کافی)

منبع قیمت: Binance Public Data Mirror (بدون نیاز به API Key)
ارسال: تلگرام

نسخه ۳ - تغییرات نسبت به نسخه قبل:
  - سیستم لاگ سیگنال (signals_log.json): هر سیگنال با فاکتورهای فعالش
    ثبت می‌شود.
  - resolve_pending_signals(): قبل از تولید سیگنال جدید، نتیجه
    سیگنال‌های قبلی (برد/باخت) با بررسی کندل‌های ۱۵دقیقه‌ای از لحظه
    صدور سیگنال تاکنون تشخیص داده می‌شود.
  - update_weights(): فقط وقتی یک فاکتور حداقل MIN_SAMPLES بار در
    سیگنال‌های بسته‌شده حضور داشته، وزنش کمی (حداکثر MAX_ADJUST_PER_CYCLE
    واحد در هر اجرا) بر اساس نرخ برد واقعی‌اش تنظیم می‌شود؛ در بازه
    [MIN_WEIGHT, MAX_WEIGHT] محدود می‌ماند تا هیچ فاکتوری کاملاً حذف یا
    مسلط نشود.
  - وزن‌ها در weights.json ذخیره و بین اجراها حفظ می‌شوند (نیاز به
    فضای ذخیره‌سازی پایدار دارد - روی سرور/VM کار می‌کند، نه در
    محیط‌های ابری موقتی/بی‌حالت).
"""

import os
import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "DOGEUSDT", "TRXUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT",
]

def env_float(name: str, default: str) -> float:
    """مثل os.environ.get ولی اگر مقدار موجود ولی رشته خالی باشد
    (مثلاً یک Secret گیت‌هاب که ساخته نشده)، همان مقدار پیش‌فرض را برمی‌گرداند."""
    val = os.environ.get(name, "").strip()
    return float(val) if val else float(default)


def env_int(name: str, default: str) -> int:
    val = os.environ.get(name, "").strip()
    return int(val) if val else int(default)


CAPITAL_USD = env_float("CAPITAL_USD", "100")
RISK_PERCENT = env_float("RISK_PERCENT", "1.5")
MARGIN_ALLOCATION_PERCENT = env_float("MARGIN_ALLOCATION_PERCENT", "20")
MAX_LEVERAGE = env_float("MAX_LEVERAGE", "10")
MIN_STOP_PERCENT = env_float("MIN_STOP_PERCENT", "0.15") / 100
RISK_REWARD = 2.0
PA_SCORE_THRESHOLD = 65

TELEGRAM_TOKEN = os.environ.get("PA_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("PA_TELEGRAM_CHAT_ID")

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

# --- تنظیمات یادگیری خودکار ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", os.path.join(BASE_DIR, "weights.json"))
SIGNALS_LOG_PATH = os.environ.get("SIGNALS_LOG_PATH", os.path.join(BASE_DIR, "signals_log.json"))
MIN_SAMPLES = env_int("MIN_SAMPLES", "15")
MAX_ADJUST_PER_CYCLE = env_float("MAX_ADJUST_PER_CYCLE", "1.5")
MIN_WEIGHT = 5.0
MAX_WEIGHT = 35.0
MAX_PENDING_AGE_HOURS = 72

OFFSET_PATH = os.environ.get("OFFSET_PATH", os.path.join(BASE_DIR, "telegram_offset.json"))
COMMAND_KEYWORDS = ("وضعیت", "status", "/status", "گزارش")
START_KEYWORDS = ("/start", "start", "شروع")

DEFAULT_WEIGHTS = {
    "structure_4h": 25.0,
    "bos_1h": 20.0,
    "choch_1h": 12.0,
    "candle_pattern": 15.0,
    "supply_demand_zone": 15.0,
    "order_block": 15.0,
    "liquidity_grab": 10.0,
}


# ---------------------------------------------------------------
# ذخیره‌سازی: وزن‌ها و لاگ سیگنال‌ها
# ---------------------------------------------------------------
def load_weights() -> dict:
    if os.path.exists(WEIGHTS_PATH):
        try:
            with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            weights = dict(DEFAULT_WEIGHTS)
            weights.update({k: v for k, v in saved.items() if k in DEFAULT_WEIGHTS})
            return weights
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights: dict):
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)


def load_signals_log() -> list:
    if os.path.exists(SIGNALS_LOG_PATH):
        try:
            with open(SIGNALS_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_signals_log(log: list):
    with open(SIGNALS_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def load_offset() -> int:
    if os.path.exists(OFFSET_PATH):
        try:
            with open(OFFSET_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("offset", 0)
        except (json.JSONDecodeError, OSError):
            return 0
    return 0


def save_offset(offset: int):
    with open(OFFSET_PATH, "w", encoding="utf-8") as f:
        json.dump({"offset": offset}, f)


# ---------------------------------------------------------------
# دریافت داده
# ---------------------------------------------------------------
def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------
# ابزارهای پرایس‌اکشن
# ---------------------------------------------------------------
def find_swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df = df.copy()
    highs, lows = df["high"], df["low"]
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    for i in range(window, len(df) - window):
        if highs.iloc[i] == highs.iloc[i - window:i + window + 1].max():
            swing_high.iloc[i] = True
        if lows.iloc[i] == lows.iloc[i - window:i + window + 1].min():
            swing_low.iloc[i] = True
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def get_last_swings(df: pd.DataFrame, n: int = 3):
    highs = df.loc[df["swing_high"], "high"].tail(n).tolist()
    lows = df.loc[df["swing_low"], "low"].tail(n).tolist()
    return highs, lows


def market_structure_trend(df: pd.DataFrame):
    df = find_swing_points(df)
    highs, lows = get_last_swings(df, 3)
    trend = "range"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            trend = "bull"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            trend = "bear"
    return trend, highs, lows


def detect_bos_choch(df: pd.DataFrame, trend: str):
    df = find_swing_points(df)
    highs, lows = get_last_swings(df, 2)
    last_close = df["close"].iloc[-1]
    if trend == "bull" and highs and last_close > highs[-1]:
        return "bull_bos"
    if trend == "bear" and lows and last_close < lows[-1]:
        return "bear_bos"
    if trend == "bull" and lows and last_close < lows[-1]:
        return "bear_choch"
    if trend == "bear" and highs and last_close > highs[-1]:
        return "bull_choch"
    return None


def detect_candle_pattern(df: pd.DataFrame):
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c["close"] - c["open"])
    rng = c["high"] - c["low"]
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]

    if c["close"] > c["open"] and p["close"] < p["open"] and c["close"] >= p["open"] and c["open"] <= p["close"]:
        return "bullish_engulfing"
    if c["close"] < c["open"] and p["close"] > p["open"] and c["open"] >= p["close"] and c["close"] <= p["open"]:
        return "bearish_engulfing"
    if rng > 0 and lower_wick > 2 * body and upper_wick < body:
        return "bullish_pin"
    if rng > 0 and upper_wick > 2 * body and lower_wick < body:
        return "bearish_pin"
    return None


def find_supply_demand_zone(df: pd.DataFrame, lookback: int = 40):
    recent = df.tail(lookback).copy()
    recent["body"] = (recent["close"] - recent["open"]).abs()
    avg_body = recent["body"].mean()
    strong = recent[recent["body"] > 1.5 * avg_body]
    if strong.empty:
        return None
    last_strong = strong.iloc[-1]
    if last_strong["close"] > last_strong["open"]:
        return ("demand", last_strong["low"], last_strong["high"])
    return ("supply", last_strong["low"], last_strong["high"])


def find_order_block(df: pd.DataFrame, lookback: int = 40):
    recent = df.tail(lookback).reset_index(drop=True).copy()
    recent["body"] = (recent["close"] - recent["open"]).abs()
    avg_body = recent["body"].mean()
    for i in range(len(recent) - 1, 0, -1):
        candle = recent.iloc[i]
        prev = recent.iloc[i - 1]
        if candle["body"] > 1.5 * avg_body:
            if candle["close"] > candle["open"] and prev["close"] < prev["open"]:
                return ("bullish_ob", prev["low"], prev["high"])
            if candle["close"] < candle["open"] and prev["close"] > prev["open"]:
                return ("bearish_ob", prev["low"], prev["high"])
    return None


def price_in_zone(price: float, zone) -> bool:
    if zone is None:
        return False
    _, lo, hi = zone
    return lo <= price <= hi


def detect_liquidity_grab(df: pd.DataFrame):
    df = find_swing_points(df)
    highs, lows = get_last_swings(df, 2)
    c = df.iloc[-1]
    if lows and c["low"] < lows[-1] and c["close"] > lows[-1]:
        return "bullish_grab"
    if highs and c["high"] > highs[-1] and c["close"] < highs[-1]:
        return "bearish_grab"
    return None


# ---------------------------------------------------------------
# تحلیل اصلی پرایس‌اکشن (چندتایم‌فریمی) - وزن‌محور و قابل یادگیری
# ---------------------------------------------------------------
def analyze_price_action(symbol: str, weights: dict):
    df4h = fetch_klines(symbol, "4h", 150)
    df1h = fetch_klines(symbol, "1h", 200)
    df15m = fetch_klines(symbol, "15m", 200)

    bull, bear = 0.0, 0.0
    reasons = []
    bull_factors, bear_factors = [], []

    trend4h, _, _ = market_structure_trend(df4h)
    if trend4h == "bull":
        bull += weights["structure_4h"]
        bull_factors.append("structure_4h")
        reasons.append("ساختار بازار (۴ساعته): سقف/کف‌های بالاتر (روند صعودی)")
    elif trend4h == "bear":
        bear += weights["structure_4h"]
        bear_factors.append("structure_4h")
        reasons.append("ساختار بازار (۴ساعته): سقف/کف‌های پایین‌تر (روند نزولی)")

    trend1h, _, _ = market_structure_trend(df1h)
    bos = detect_bos_choch(df1h, trend1h if trend1h != "range" else trend4h)
    if bos == "bull_bos":
        bull += weights["bos_1h"]
        bull_factors.append("bos_1h")
        reasons.append("شکست ساختار صعودی (BOS) در ۱ساعته - ادامه روند")
    elif bos == "bear_bos":
        bear += weights["bos_1h"]
        bear_factors.append("bos_1h")
        reasons.append("شکست ساختار نزولی (BOS) در ۱ساعته - ادامه روند")
    elif bos == "bull_choch":
        bull += weights["choch_1h"]
        bull_factors.append("choch_1h")
        reasons.append("نشانه تغییر روند به صعودی (CHoCH) در ۱ساعته")
    elif bos == "bear_choch":
        bear += weights["choch_1h"]
        bear_factors.append("choch_1h")
        reasons.append("نشانه تغییر روند به نزولی (CHoCH) در ۱ساعته")

    candle_1h = detect_candle_pattern(df1h)
    candle_15m = detect_candle_pattern(df15m)
    if candle_1h in ("bullish_engulfing", "bullish_pin") or candle_15m in ("bullish_engulfing", "bullish_pin"):
        bull += weights["candle_pattern"]
        bull_factors.append("candle_pattern")
        pat = candle_15m if candle_15m in ("bullish_engulfing", "bullish_pin") else candle_1h
        reasons.append(f"الگوی کندلی صعودی شناسایی شد ({pat})")
    elif candle_1h in ("bearish_engulfing", "bearish_pin") or candle_15m in ("bearish_engulfing", "bearish_pin"):
        bear += weights["candle_pattern"]
        bear_factors.append("candle_pattern")
        pat = candle_15m if candle_15m in ("bearish_engulfing", "bearish_pin") else candle_1h
        reasons.append(f"الگوی کندلی نزولی شناسایی شد ({pat})")

    zone = find_supply_demand_zone(df1h)
    price = df15m["close"].iloc[-1]
    if zone and price_in_zone(price, zone):
        if zone[0] == "demand":
            bull += weights["supply_demand_zone"]
            bull_factors.append("supply_demand_zone")
            reasons.append("قیمت داخل منطقه تقاضا (Demand Zone)")
        else:
            bear += weights["supply_demand_zone"]
            bear_factors.append("supply_demand_zone")
            reasons.append("قیمت داخل منطقه عرضه (Supply Zone)")

    ob = find_order_block(df1h)
    if ob and price_in_zone(price, ob):
        if ob[0] == "bullish_ob":
            bull += weights["order_block"]
            bull_factors.append("order_block")
            reasons.append("قیمت داخل Order Block صعودی (ورود احتمالی پول هوشمند)")
        else:
            bear += weights["order_block"]
            bear_factors.append("order_block")
            reasons.append("قیمت داخل Order Block نزولی (ورود احتمالی پول هوشمند)")

    grab_1h = detect_liquidity_grab(df1h)
    grab_15m = detect_liquidity_grab(df15m)
    if grab_1h == "bullish_grab" or grab_15m == "bullish_grab":
        bull += weights["liquidity_grab"]
        bull_factors.append("liquidity_grab")
        reasons.append("جمع‌آوری نقدینگی زیر کف قبلی و بازگشت (Liquidity Grab صعودی)")
    elif grab_1h == "bearish_grab" or grab_15m == "bearish_grab":
        bear += weights["liquidity_grab"]
        bear_factors.append("liquidity_grab")
        reasons.append("جمع‌آوری نقدینگی بالای سقف قبلی و بازگشت (Liquidity Grab نزولی)")

    dominant = max(bull, bear)
    opposite = min(bull, bear)
    if dominant > 0:
        conflict_ratio = opposite / dominant
        if conflict_ratio >= 0.4:
            dominant *= 0.85
            reasons.append("⚠️ برخی سیگنال‌ها در تناقض بودند؛ امتیاز اطمینان کاهش یافت")
        elif conflict_ratio == 0 and dominant >= 60:
            dominant = min(100, dominant * 1.05)

    if bull >= bear:
        confidence = dominant
        direction = "BUY" if bull > bear else "NEUTRAL"
        contributing_factors = bull_factors
    else:
        confidence = dominant
        direction = "SELL"
        contributing_factors = bear_factors

    if confidence < PA_SCORE_THRESHOLD or direction == "NEUTRAL":
        return None

    df1h_sw = find_swing_points(df1h)
    highs1h, lows1h = get_last_swings(df1h_sw, 3)
    atr1h = atr(df1h).iloc[-1]
    entry = price

    if direction == "BUY":
        structure_sl = lows1h[-1] if lows1h else entry - 1.5 * atr1h
        sl = min(structure_sl, entry - 0.5 * atr1h) - 0.25 * atr1h
        tp = entry + (entry - sl) * RISK_REWARD
    else:
        structure_sl = highs1h[-1] if highs1h else entry + 1.5 * atr1h
        sl = max(structure_sl, entry + 0.5 * atr1h) + 0.25 * atr1h
        tp = entry - (sl - entry) * RISK_REWARD

    stop_distance_pct = abs(entry - sl) / entry
    if stop_distance_pct < MIN_STOP_PERCENT:
        return None

    risk_amount_usd = CAPITAL_USD * RISK_PERCENT / 100
    margin_usd = CAPITAL_USD * MARGIN_ALLOCATION_PERCENT / 100
    position_usd = risk_amount_usd / stop_distance_pct
    required_leverage = position_usd / margin_usd

    if required_leverage < 1:
        leverage = 1.0
        position_usd = margin_usd
    elif required_leverage > MAX_LEVERAGE:
        return None
    else:
        leverage = required_leverage

    actual_risk_usd = position_usd * stop_distance_pct
    tp_distance_pct = abs(tp - entry) / entry
    potential_profit_usd = position_usd * tp_distance_pct
    actual_rr = tp_distance_pct / stop_distance_pct

    return {
        "id": str(uuid.uuid4()),
        "symbol": symbol,
        "direction": direction,
        "confidence": round(confidence),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "stop_distance_pct": stop_distance_pct * 100,
        "actual_rr": actual_rr,
        "reasons": reasons,
        "factors": contributing_factors,
        "margin_usd": margin_usd,
        "leverage": round(leverage, 1),
        "position_usd": position_usd,
        "risk_usd": actual_risk_usd,
        "profit_usd": potential_profit_usd,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }


# ---------------------------------------------------------------
# یادگیری خودکار: بررسی نتیجه سیگنال‌های قبلی و تنظیم وزن‌ها
# ---------------------------------------------------------------
def resolve_pending_signals(log: list) -> bool:
    changed = False
    now = datetime.now(timezone.utc)
    for sig in log:
        if sig.get("status") != "pending":
            continue
        signal_time = datetime.fromisoformat(sig["timestamp"])
        age_hours = (now - signal_time).total_seconds() / 3600

        try:
            df = fetch_klines(sig["symbol"], "15m", 500)
        except Exception:
            continue

        df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        relevant = df[df["open_time_dt"] >= signal_time]
        if relevant.empty:
            if age_hours > MAX_PENDING_AGE_HOURS:
                sig["status"] = "expired"
                sig["closed_at"] = now.isoformat()
                changed = True
            continue

        result = None
        for _, row in relevant.iterrows():
            if sig["direction"] == "BUY":
                hit_tp = row["high"] >= sig["tp"]
                hit_sl = row["low"] <= sig["sl"]
            else:
                hit_tp = row["low"] <= sig["tp"]
                hit_sl = row["high"] >= sig["sl"]
            if hit_tp and hit_sl:
                result = "loss"
                break
            if hit_tp:
                result = "win"
                break
            if hit_sl:
                result = "loss"
                break

        if result:
            sig["status"] = result
            sig["closed_at"] = now.isoformat()
            changed = True
        elif age_hours > MAX_PENDING_AGE_HOURS:
            sig["status"] = "expired"
            sig["closed_at"] = now.isoformat()
            changed = True

    return changed


def update_weights(log: list, weights: dict) -> bool:
    stats = {k: {"win": 0, "total": 0} for k in weights}
    for sig in log:
        if sig.get("status") not in ("win", "loss"):
            continue
        for f in sig.get("factors", []):
            if f in stats:
                stats[f]["total"] += 1
                if sig["status"] == "win":
                    stats[f]["win"] += 1

    changed = False
    for factor, s in stats.items():
        if s["total"] < MIN_SAMPLES:
            continue
        win_rate = s["win"] / s["total"]
        raw_delta = (win_rate - 0.5) * 2 * MAX_ADJUST_PER_CYCLE
        delta = max(-MAX_ADJUST_PER_CYCLE, min(MAX_ADJUST_PER_CYCLE, raw_delta))
        new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, weights[factor] + delta))
        if abs(new_weight - weights[factor]) >= 0.1:
            weights[factor] = round(new_weight, 1)
            changed = True

    return changed


def weights_summary_text(weights: dict, log: list) -> str:
    stats = {k: {"win": 0, "total": 0} for k in weights}
    for sig in log:
        if sig.get("status") not in ("win", "loss"):
            continue
        for f in sig.get("factors", []):
            if f in stats:
                stats[f]["total"] += 1
                if sig["status"] == "win":
                    stats[f]["win"] += 1
    lines = ["<b>📊 وضعیت فعلی وزن فاکتورها (یادگیری خودکار)</b>"]
    for factor, w in weights.items():
        s = stats[factor]
        if s["total"] > 0:
            wr = 100 * s["win"] / s["total"]
            lines.append(f"• {factor}: وزن {w} | نرخ برد {wr:.0f}٪ ({s['total']} نمونه)")
        else:
            lines.append(f"• {factor}: وزن {w} | هنوز نمونه کافی نیست")
    return "\n".join(lines)


# ---------------------------------------------------------------
# ارسال به تلگرام
# ---------------------------------------------------------------
def get_telegram_updates(offset: int):
    if not TELEGRAM_TOKEN:
        return []
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": 0}
    if offset:
        params["offset"] = offset
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("result", [])


def check_and_reply_commands(weights: dict, log: list):
    """
    یک‌بار در هر اجرا، پیام‌های جدید تلگرام (از آخرین offset ذخیره‌شده)
    را می‌خواند. اگر پیامی از چت مجاز (TELEGRAM_CHAT_ID) حاوی یکی از
    COMMAND_KEYWORDS باشد، خلاصه وضعیت وزن‌ها/سیگنال‌ها را پاسخ می‌دهد.
    چون این تابع فقط زمانی که کل اسکریپت اجرا می‌شود (مثلاً هر ۱۵ دقیقه
    توسط GitHub Actions) صدا زده می‌شود، پاسخ فوری نیست - تا اجرای بعدی
    طول می‌کشد.
    """
    offset = load_offset()
    try:
        updates = get_telegram_updates(offset)
    except Exception as e:
        print(f"خطا در دریافت پیام‌های تلگرام: {e}")
        return

    max_update_id = offset - 1
    for update in updates:
        max_update_id = max(max_update_id, update.get("update_id", 0))
        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = (message.get("text") or "").strip().lower()

        if TELEGRAM_CHAT_ID and chat_id != str(TELEGRAM_CHAT_ID):
            continue  # فقط به چت مجاز پاسخ بده

        if text in START_KEYWORDS:
            send_telegram_message(
                "👋 سلام! ربات پرایس‌اکشن روشن و متصله.\n"
                "هر وقت یک ستاپ باکیفیت پیدا کنه، اینجا سیگنال می‌فرستم.\n"
                "برای دیدن آمار سیگنال‌ها و وزن فاکتورها، کلمه «وضعیت» یا «گزارش» رو بفرست."
            )
            continue

        if any(kw in text for kw in COMMAND_KEYWORDS):
            wins = sum(1 for s in log if s.get("status") == "win")
            losses = sum(1 for s in log if s.get("status") == "loss")
            pending = sum(1 for s in log if s.get("status") == "pending")
            total_closed = wins + losses
            win_rate_txt = f"{100 * wins / total_closed:.0f}٪" if total_closed else "هنوز داده کافی نیست"
            reply = (
                f"📈 از {total_closed} سیگنال بسته‌شده: <b>{wins} تی‌پی</b> | <b>{losses} استاپ</b>\n"
                f"نرخ برد کلی: <b>{win_rate_txt}</b>\n"
                f"سیگنال‌های در انتظار نتیجه: {pending}\n\n"
                + weights_summary_text(weights, log)
            )
            send_telegram_message(reply)

    if max_update_id >= offset:
        save_offset(max_update_id + 1)


def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("توکن یا چت آیدی تلگرام تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    resp = requests.post(url, data=payload, timeout=15)
    if resp.status_code != 200:
        print("خطا در ارسال به تلگرام:", resp.text)


def format_message(r) -> str:
    direction_fa = "خرید (BUY) 🟢" if r["direction"] == "BUY" else "فروش (SELL) 🔴"
    lines = [
        f"<b>🕯️ سیگنال پرایس‌اکشن | {r['symbol']}</b>",
        f"امتیاز اطمینان: <b>{r['confidence']} / 100</b>",
        f"جهت: <b>{direction_fa}</b>",
        "",
        f"قیمت ورود: {r['entry']:.5f}",
        f"🎯 حد سود (TP): {r['tp']:.5f}",
        f"🛑 حد ضرر (SL): {r['sl']:.5f}",
        f"فاصله استاپ: {r['stop_distance_pct']:.2f}٪ | نسبت ریسک‌به‌ریوارد واقعی: {r['actual_rr']:.2f}",
        "",
        "<b>💵 مدیریت سرمایه (بر اساس سرمایه $" + f"{CAPITAL_USD:.0f})</b>",
        f"مارجین این معامله: <b>${r['margin_usd']:.2f}</b>",
        f"اهرم پیشنهادی: <b>{r['leverage']}x</b>",
        f"حجم کل پوزیشن: ${r['position_usd']:.2f}",
        f"ریسک این معامله: <b>${r['risk_usd']:.2f}</b> (≈{RISK_PERCENT}٪ سرمایه)",
        f"سود هدف: <b>${r['profit_usd']:.2f}</b>",
        "",
        "<i>دلایل تحلیل پرایس‌اکشن (۴ساعته → ۱ساعته → ۱۵دقیقه):</i>",
    ]
    lines += [f"• {x}" for x in r["reasons"]]
    lines.append(
        "\n⚠️ فقط با همین حجم و اهرم پیشنهادی وارد شو. نتیجه این سیگنال "
        "به‌صورت خودکار بررسی و در تنظیم وزن تحلیل‌های آینده استفاده می‌شود. "
        "صرفاً تحلیل خودکار است، نه تضمین سود."
    )
    return "\n".join(lines)


def main():
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"WEIGHTS_PATH: {WEIGHTS_PATH}")
    print(f"SIGNALS_LOG_PATH: {SIGNALS_LOG_PATH}")
    print(f"OFFSET_PATH: {OFFSET_PATH}")

    weights = load_weights()
    log = load_signals_log()

    # ۰) اول ببین کاربر پیام/دستوری فرستاده (مثلاً "وضعیت") و جواب بده
    check_and_reply_commands(weights, log)

    if resolve_pending_signals(log):
        save_signals_log(log)
        print("نتیجه سیگنال‌های قبلی به‌روزرسانی شد.")

    if update_weights(log, weights):
        save_weights(weights)
        print("وزن فاکتورها بر اساس عملکرد واقعی تنظیم شد:")
        print(json.dumps(weights, ensure_ascii=False, indent=2))

    any_signal = False
    for symbol in SYMBOLS:
        try:
            result = analyze_price_action(symbol, weights)
            if result:
                send_telegram_message(format_message(result))
                log.append(result)
                any_signal = True
                print(f"سیگنال پرایس‌اکشن {symbol} ارسال شد (امتیاز {result['confidence']}).")
            else:
                print(f"{symbol}: شرایط پرایس‌اکشن با کیفیت کافی یا مدیریت ریسک قابل‌قبول پیدا نشد.")
        except Exception as e:
            print(f"خطا در تحلیل {symbol}: {e}")

    save_signals_log(log)

    if not any_signal:
        print("در این اجرا هیچ ارزی ستاپ پرایس‌اکشن با کیفیت نداشت.")


if __name__ == "__main__":
    main()
