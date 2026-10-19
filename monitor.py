"""チャート通知プログラム

- ドル円・原油・ゴールド：5分足ボリンジャーバンド(20, 3σ)の %B が
  0.05以下（−3σの手前）/ 0.95以上（+3σの手前）になったら通知
- 日経：日足・1時間足の RSI(14) が 70以上 / 30以下になったら通知

同じ状態のあいだは何度も通知しない。いったん普通の範囲に戻ったら、
次にまた条件に来たときに通知する。
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

JST = timezone(timedelta(hours=9))
STATE_FILE = Path(__file__).with_name("state.json")

# ===== ここを変えれば設定を変えられます =====
BB_TARGETS = [
    # (表示名, Yahooの記号)
    ("ドル円", "USDJPY=X"),
    ("原油(WTI)", "CL=F"),
    ("ゴールド", "GC=F"),
]
BB_PERIOD = 20
BB_SIGMA = 3.0
BB_LOW = 0.05     # これ以下で「−3σの手前」
BB_HIGH = 0.95    # これ以上で「+3σの手前」
BB_RESET_LOW = 0.20   # ここより上に戻ったら、次の通知の準備
BB_RESET_HIGH = 0.80

RSI_TARGETS = [
    # (表示名, Yahooの記号, 足, 足の名前, 取ってくる期間)
    ("日経", "NIY=F", "1d", "日足", "1y"),
    ("日経", "NIY=F", "60m", "1時間足", "1mo"),
]
RSI_PERIOD = 14
RSI_HIGH = 70
RSI_LOW = 30
RSI_RESET_HIGH = 65
RSI_RESET_LOW = 35

# データがこれより古いときは、市場が閉まっているとみなして何もしない
MAX_AGE = {"5m": timedelta(minutes=30), "60m": timedelta(hours=3), "1d": timedelta(days=5)}
# ============================================


def fetch(symbol, interval, period):
    import yfinance as yf

    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # 列が二重になっている場合
        close = close.iloc[:, 0]
    return close.dropna()


def percent_b(close, period=BB_PERIOD, sigma=BB_SIGMA):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = mid + sigma * sd
    lower = mid - sigma * sd
    return (close - lower) / (upper - lower)


def rsi(close, period=RSI_PERIOD):
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def next_zone(prev, value, low, high, reset_low, reset_high):
    """今の状態（low / high / none）を決める。戻りに少し幅を持たせて、
    境目で何度も通知が来ないようにする。"""
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    if prev == "low" and value < reset_low:
        return "low"
    if prev == "high" and value > reset_high:
        return "high"
    return "none"


def is_fresh(close, interval, now):
    last = close.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    return now - last.to_pydatetime() <= MAX_AGE[interval]


def notify(title, message, priority=4, tags=None):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print(f"[通知先なし] {title} / {message}")
        return
    r = requests.post(
        "https://ntfy.sh/",
        data=json.dumps({
            "topic": topic,
            "title": title,
            "message": message,
            "priority": priority,
            "tags": tags or [],
        }, ensure_ascii=False).encode("utf-8"),
        timeout=20,
    )
    r.raise_for_status()


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def check_bb(state, now, fetcher, alerts, report):
    for name, symbol in BB_TARGETS:
        key = f"bb:{symbol}"
        try:
            close = fetcher(symbol, "5m", "5d")
        except Exception as e:  # 1つ失敗しても他は続ける
            print(f"{name}: 取得失敗 {e}")
            continue
        if close is None or len(close) < BB_PERIOD + 1:
            print(f"{name}: データ不足")
            continue
        pb = float(percent_b(close).iloc[-1])
        price = float(close.iloc[-1])
        report.append(f"{name} 5分足 %B {pb:.2f}（{price:g}）")
        if not is_fresh(close, "5m", now):
            print(f"{name}: 市場が閉まっているのでスキップ")
            continue
        prev = state.get(key, "none")
        zone = next_zone(prev, pb, BB_LOW, BB_HIGH, BB_RESET_LOW, BB_RESET_HIGH)
        state[key] = zone
        if zone != prev and zone != "none":
            side = "−3σの手前" if zone == "low" else "+3σの手前"
            alerts.append((f"{name} {side}", f"5分足 %B {pb:.2f}／価格 {price:g}",
                           ["chart_with_downwards_trend" if zone == "low" else "chart_with_upwards_trend"]))


def check_rsi(state, now, fetcher, alerts, report):
    for name, symbol, interval, label, period in RSI_TARGETS:
        key = f"rsi:{symbol}:{interval}"
        try:
            close = fetcher(symbol, interval, period)
        except Exception as e:
            print(f"{name} {label}: 取得失敗 {e}")
            continue
        if close is None or len(close) < RSI_PERIOD + 2:
            print(f"{name} {label}: データ不足")
            continue
        value = float(rsi(close).iloc[-1])
        price = float(close.iloc[-1])
        report.append(f"{name} {label} RSI {value:.1f}（{price:g}）")
        if not is_fresh(close, interval, now):
            print(f"{name} {label}: 市場が閉まっているのでスキップ")
            continue
        prev = state.get(key, "none")
        zone = next_zone(prev, value, RSI_LOW, RSI_HIGH, RSI_RESET_LOW, RSI_RESET_HIGH)
        state[key] = zone
        if zone != prev and zone != "none":
            side = f"RSI {RSI_HIGH}以上" if zone == "high" else f"RSI {RSI_LOW}以下"
            alerts.append((f"{name} {label} {side}", f"RSI {value:.1f}／価格 {price:g}",
                           ["bell"]))


def run(fetcher=fetch, now=None, test=False):
    now = now or datetime.now(timezone.utc)
    state = load_state()
    before = json.dumps(state, sort_keys=True)
    alerts, report = [], []

    check_bb(state, now, fetcher, alerts, report)
    check_rsi(state, now, fetcher, alerts, report)

    for title, message, tags in alerts:
        print(f"通知: {title} / {message}")
        notify(title, message, tags=tags)

    if test:
        stamp = now.astimezone(JST).strftime("%m/%d %H:%M")
        notify(f"テスト通知（{stamp}）", "\n".join(report) or "データを取れませんでした",
               priority=3, tags=["white_check_mark"])

    print("\n".join(report))
    if json.dumps(state, sort_keys=True) != before:
        save_state(state)
    return alerts, state


if __name__ == "__main__":
    run(test="--test" in sys.argv)
