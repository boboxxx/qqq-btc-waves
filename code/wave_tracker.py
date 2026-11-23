"""Follow the current Elliott-wave count for QQQ and BTC as new daily closes arrive.

The count itself (which pivots are settled, which leg is in progress) lives in code/waves/wave_count.json.
From that point on everything is mechanical and recomputed from closes on every run:

* the leg in progress runs to its most extreme close;
* "possibly over" once price comes back `turn` (QQQ 5%, BTC 12%) from that extreme;
* "confirmed over" once a close crosses the leg's last internal counter-swing (a `swing` zigzag pivot,
  QQQ 5%, BTC 6%) or, if it has none, the leg's own starting price; the next leg (A -> B -> C) then starts;
* A and C legs of this correction are invalid above the top; a B leg above the top is flagged for a recount.

Events (leg changes, level crossings, new extremes) are taken from settled daily closes only, so an intraday
spike cannot trigger them; the page also shows how things stand at the live price. Outside regular US hours the
live QQQ price is estimated from the Nasdaq-100 future (NQ=F): last close x (future now / future at that close).

Usage:
  python code/wave_tracker.py                 # one update: status JSON, event log, page
  python code/wave_tracker.py --loop 900      # repeat every 15 minutes
  python code/wave_tracker.py --email         # also email key events (uses the signal-alert SMTP settings)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
import time
from datetime import datetime, timezone
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "code" / "waves" / "wave_count.json"
TEMPLATE = ROOT / "code" / "waves" / "waves_template.html"
DEFAULT_OUT = ROOT / "data" / "waves"

NYSE_HOLIDAYS = pd.to_datetime([
    "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25",
])
NYSE_DAY = pd.offsets.CustomBusinessDay(holidays=NYSE_HOLIDAYS)
DOWN_LEGS = {"A", "C"}  # this is a correction of an advance: A and C fall, B rises
LETTERS = "abcdefghij"


# ----------------------------------------------------------------------------------------- data
def fetch_daily(ticker: str, start: str) -> tuple[pd.Series, dict]:
    """Daily closes from Yahoo. Returns (closes including today's live bar, info about the live bar)."""
    last_error = None
    for attempt in range(3):
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                             params={"interval": "1d", "period1": int(pd.Timestamp(start).timestamp()), "period2": int(time.time()), "includePrePost": "false"},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            break
        except Exception as exc:  # noqa: BLE001 - network hiccups are retried, then re-raised
            last_error = exc
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"{ticker}: download failed: {last_error}")
    meta, tz = res["meta"], res["meta"]["exchangeTimezoneName"]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=idx, dtype="float64").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    now = pd.Timestamp.now(tz=tz).tz_localize(None)
    last = s.index[-1]
    if tz == "UTC":  # crypto: the daily bar closes at 00:00 UTC
        provisional = last.date() == now.date()
    else:
        provisional = last.date() == now.date() and now.time() < dtime(16, 15)
    price = float(meta.get("regularMarketPrice") or s.iloc[-1])
    if provisional:
        s.iloc[-1] = price
    stamp = pd.Timestamp(int(meta.get("regularMarketTime") or time.time()), unit="s", tz="UTC").tz_convert(tz)
    return s, {"provisional": bool(provisional), "price": price, "price_time": stamp.strftime("%Y-%m-%d %H:%M"), "tz": tz,
               "label": "盘中" if provisional else "收盘", "source": ticker}


def fetch_intraday(ticker: str) -> tuple[pd.Series, float, pd.Timestamp]:
    """Five-minute closes (New York time, bar start) for the last ten days, plus the latest price and its time."""
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", params={"interval": "5m", "range": "10d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ny = "America/New_York"
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(ny).tz_localize(None)
    bars = pd.Series(res["indicators"]["quote"][0]["close"], index=idx, dtype="float64").dropna().sort_index()
    meta = res["meta"]
    t = pd.Timestamp(int(meta["regularMarketTime"]), unit="s", tz="UTC").tz_convert(ny).tz_localize(None)
    return bars, float(meta["regularMarketPrice"]), t


def overnight_estimate(s: pd.Series, info: dict, proxy: str, now: pd.Timestamp | None = None) -> tuple[pd.Series, dict]:
    """Outside regular US hours, estimate QQQ from the Nasdaq-100 future (it trades almost round the clock):
    last QQQ close x (future now / future at that close). The estimate becomes a provisional bar for the next
    session, so it moves the live view only; settled closes, events and emails never see it."""
    ny = "America/New_York"
    now = pd.Timestamp.now(tz=ny).tz_localize(None) if now is None else now
    if NYSE_DAY.is_on_offset(now.normalize()) and dtime(9, 30) <= now.time() < dtime(16, 15):
        return s, info  # the market is open: QQQ's own price
    if info["provisional"]:  # a bar for today before the open is a placeholder, not a close
        s = s.iloc[:-1]
    close_day = s.index[-1]
    close_at = close_day + pd.Timedelta(hours=16)
    settled_info = {"provisional": False, "price": round(float(s.iloc[-1]), 2), "price_time": close_at.strftime("%Y-%m-%d %H:%M"),
                    "tz": ny, "label": "收盘", "source": info.get("source", "QQQ")}
    try:
        bars, fut, fut_t = fetch_intraday(proxy)
    except Exception as exc:  # noqa: BLE001 - without the future the page simply shows the last close
        print(f"{proxy}: download failed, showing the last close: {exc}", file=sys.stderr)
        return s, settled_info
    near = bars[(bars.index < close_at) & (bars.index >= close_at - pd.Timedelta(minutes=30))]
    if near.empty or fut_t <= close_at + pd.Timedelta(minutes=1):
        return s, settled_info  # no future price at the close, or nothing traded since
    ref = float(near.iloc[-1])  # the bar that ends at 16:00
    est = float(s.iloc[-1]) * fut / ref
    s = pd.concat([s, pd.Series([est], index=[close_day + NYSE_DAY])])
    return s, {"provisional": True, "price": round(est, 2), "price_time": fut_t.strftime("%Y-%m-%d %H:%M"), "tz": ny,
               "label": "期货估算", "source": proxy,
               "proxy": {"symbol": proxy, "price": round(fut, 2), "ref": round(ref, 2), "ref_time": close_at.strftime("%Y-%m-%d %H:%M"),
                         "move": round(fut / ref - 1, 5), "close": round(float(s.iloc[-2]), 2)}}


class Clock:
    """Counts time in trading sessions (QQQ) or calendar days (BTC), and projects forward."""

    def __init__(self, index: pd.DatetimeIndex, unit: str):
        self.index, self.unit = index, unit

    def dist(self, d0: pd.Timestamp, d1: pd.Timestamp) -> int:
        if self.unit == "days":
            return int((d1 - d0).days)
        return int(self.index.get_loc(d1) - self.index.get_loc(d0))

    def add(self, d0: pd.Timestamp, n: float) -> pd.Timestamp:
        if self.unit == "days":
            return d0 + pd.Timedelta(days=round(n))
        k = self.index.get_loc(d0) + int(round(n))
        if k < len(self.index):
            return self.index[k]
        return self.index[-1] + (k - len(self.index) + 1) * NYSE_DAY


# ----------------------------------------------------------------------------------------- swings
def zigzag(v: np.ndarray, th: float, first: str) -> list[tuple[int, str]]:
    """Confirmed swing pivots of v; v[0] is taken as a pivot of type `first` ('hi' or 'lo')."""
    piv, trend, ext = [(0, first)], (-1 if first == "hi" else 1), 0
    for i in range(1, len(v)):
        if trend < 0:
            if v[i] < v[ext]:
                ext = i
            elif v[i] >= v[ext] * (1 + th):
                piv.append((ext, "lo")); trend, ext = 1, i
        else:
            if v[i] > v[ext]:
                ext = i
            elif v[i] <= v[ext] * (1 - th):
                piv.append((ext, "hi")); trend, ext = -1, i
    return piv


def internal_pivots(v: np.ndarray, down: bool, th: float) -> list[tuple[int, str]]:
    """Swing pivots strictly between the leg start (v[0]) and the leg extreme (v[-1])."""
    return [p for p in zigzag(v, th, "hi" if down else "lo")[1:] if 0 < p[0] < len(v) - 1]


def confirm_point(v: np.ndarray, down: bool, th: float) -> int:
    """Index of the level whose break confirms the leg is over: the last internal counter-swing, else the leg start."""
    counter = [p[0] for p in internal_pivots(v, down, th) if p[1] == ("hi" if down else "lo")]
    return counter[-1] if counter else 0


def scan_leg(v: np.ndarray, down: bool, swing: float) -> dict:
    """Walk the leg forward in time; stop at the first close that confirms it ended."""
    ext = 0
    for i in range(1, len(v)):
        if (v[i] < v[ext]) if down else (v[i] > v[ext]):
            ext = i
            continue
        if ext == 0:
            continue
        c = confirm_point(v[:ext + 1], down, swing)
        if (v[i] > v[c]) if down else (v[i] < v[c]):
            return {"ext": ext, "confirmed_at": i, "confirm": c}
    return {"ext": ext, "confirmed_at": None, "confirm": confirm_point(v[:ext + 1], down, swing) if ext else None}


# ----------------------------------------------------------------------------------------- the count
def evaluate(s: pd.Series, cfg: dict) -> dict:
    """Where the correction stands on closes s. Pure function of the closes and the config."""
    clock = Clock(s.index, cfg["unit"])
    legs = cfg["legs"]
    pivots = [{"date": pd.Timestamp(d), "label": lab, "price": float(s.loc[pd.Timestamp(d)])} for d, lab in cfg["confirmed"]]
    top = pivots[0]
    k = cfg["current_leg"]
    flags = []
    while True:
        if k >= len(legs):
            cur = None
            break
        leg, down = legs[k], legs[k] in DOWN_LEGS
        i0 = s.index.get_loc(pivots[-1]["date"])
        v = s.to_numpy()[i0:]
        res = scan_leg(v, down, cfg["swing"])
        conf_is_start = res["confirm"] == 0
        if res["confirmed_at"] is not None and not (leg == "A" and conf_is_start):
            ext_d = s.index[i0 + res["ext"]]
            pivots.append({"date": ext_d, "label": leg, "price": float(v[res["ext"]]),
                           "confirmed_on": str(s.index[i0 + res["confirmed_at"]].date()), "confirm_level": float(v[res["confirm"]])})
            if leg == "C":
                a_low = next(p["price"] for p in pivots if p["label"] == "A")
                if v[res["ext"]] > a_low:
                    flags.append("C 浪没有跌破 A 浪低点就结束了（截断），这套数法要重新检查")
            k += 1
            continue
        cur = {"leg": leg, "down": down, "i0": i0, "v": v, "res": res}
        break

    last_d, last_p = s.index[-1], float(s.iloc[-1])
    out = {"pivots": [], "flags": flags, "last": [str(last_d.date()), round(last_p, 2)], "top": [str(top["date"].date()), round(top["price"], 2)]}
    for p in pivots:
        out["pivots"].append({"date": str(p["date"].date()), "label": p["label"], "price": round(p["price"], 2), **({"confirmed_on": p["confirmed_on"]} if "confirmed_on" in p else {})})

    # invalidation of the whole count: back above the top while in A or C
    if cur and cur["leg"] in DOWN_LEGS and last_p > top["price"]:
        flags.append(f"收盘高于 5 浪顶 {top['price']:,.2f}：主数法失效，需要重新数")
    if cur and cur["leg"] == "B" and s.loc[pivots[-1]["date"]:].max() > top["price"]:
        flags.append(f"B 浪超过了 5 浪顶 {top['price']:,.2f}：可能是扩张平台，也可能上涨还没结束，需要重新数")

    if cur is None:
        out.update({"leg": None, "state": "done", "summary": "A-B-C 调整已确认走完。之后要重新数：新一轮推动浪，或者更复杂的调整。"})
        return out

    leg, down, i0, v, res = cur["leg"], cur["down"], cur["i0"], cur["v"], cur["res"]
    start = pivots[-1]
    ext_i = res["ext"]
    ext_d, ext_p = s.index[i0 + ext_i], float(v[ext_i])
    sign = -1 if down else 1
    after = v[ext_i + 1:]
    turn_level = ext_p * (1 + cfg["turn"]) if down else ext_p * (1 - cfg["turn"])
    turned = bool(len(after)) and bool(((after >= turn_level) if down else (after <= turn_level)).any())
    conf_i = res["confirm"]
    conf_level = float(v[conf_i]) if conf_i is not None else None
    conf_date = str(s.index[i0 + conf_i].date()) if conf_i is not None else None
    conf_is_start = conf_i == 0
    move = ext_p / start["price"] - 1

    # internal swings of the leg in progress, labelled a, b, ...
    sub = []
    if ext_i > 0:
        for j, (ii, kind) in enumerate(internal_pivots(v[:ext_i + 1], down, cfg["swing"])):
            sub.append({"date": str(s.index[i0 + ii].date()), "label": LETTERS[j], "price": round(float(v[ii]), 2), "kind": kind})
    cur_sub = LETTERS[len(sub)] if sub else None

    targets, marks = [], []
    a_piv = next((p for p in pivots if p["label"] == "A"), None)
    if leg == "A" and len(sub) >= 2:
        a_len, a_dur = start["price"] - sub[0]["price"], clock.dist(start["date"], pd.Timestamp(sub[0]["date"]))
        b_p, b_d = sub[1]["price"], pd.Timestamp(sub[1]["date"])
        for r in (1.0, 1.618):
            targets.append([round(b_p - r * a_len, 2), "c 段 = a 段" if r == 1 else f"c 段 = {r} × a 段"])
        for r in cfg["time_ratios"]["A"]:
            marks.append([str(clock.add(b_d, r * a_dur).date()), "c=a 时间位" if r == 1 else f"c={r}a 时间位", "c=a" if r == 1 else f"c={r}a"])
    if leg == "B" and a_piv:
        a_len, a_dur = top["price"] - a_piv["price"], clock.dist(top["date"], a_piv["date"])
        for r in (0.382, 0.5, 0.618, 0.786):
            targets.append([round(a_piv["price"] + r * a_len, 2), f"B 反弹 A 浪的 {r:.1%}".replace(".0%", "%")])
        for r in cfg["time_ratios"]["B"]:
            marks.append([str(clock.add(a_piv["date"], r * a_dur).date()), f"B={r}A 时间位", f"B={r}A"])
    if leg == "C" and a_piv:
        b_piv = pivots[-1]
        a_dur = clock.dist(top["date"], a_piv["date"])
        for r in cfg["price_ratios"]["C"]:
            if cfg["measure"] == "percent":
                tgt = b_piv["price"] * (1 - r * (1 - a_piv["price"] / top["price"]))
            else:
                tgt = b_piv["price"] - r * (top["price"] - a_piv["price"])
            if tgt > 0:
                targets.append([round(tgt, 2), ("C = A" if r == 1 else f"C = {r} × A") + ("（按幅度）" if cfg["measure"] == "percent" else "")])
        for r in cfg["time_ratios"]["C"]:
            marks.append([str(clock.add(b_piv["date"], r * a_dur).date()), "C=A 时间位" if r == 1 else f"C={r}A 时间位", "C=A" if r == 1 else f"C={r}A"])
    for d, lab, only, short in cfg.get("dates", []):
        if only in (None, leg):
            marks.append([d, lab, short])
    marks.sort()

    name = f"{leg} 浪"
    conf_what = None
    if conf_i is None or (leg == "A" and conf_is_start):
        conf_text = None  # no internal counter-swing yet (or only the top): only the tentative signal applies
    else:
        conf_what = f"{name}内最后一个反弹高点" if down else f"{name}内最后一个回落低点"
        if conf_is_start:
            conf_what = f"{start['label']} 浪{'顶' if down else '底'}，即{name}起点"
        conf_text = f"收盘{'站上' if down else '跌破'} {conf_level:,.2f}（{conf_what}，{conf_date}）"
    state = "turned" if turned else "running"
    if state == "running":
        summary = f"{name}进行中" + (f"，内部第 {cur_sub} 段" if cur_sub else "") + f"。目前{'最低' if down else '最高'}收盘 {ext_p:,.2f}（{ext_d.date()}），比起点{'下跌' if down else '上涨'} {abs(move):.1%}。"
    else:
        summary = f"{name}可能已经结束：从 {ext_p:,.2f}（{ext_d.date()}）{'反弹' if down else '回落'}超过 {cfg['turn']:.0%}，等待确认。"
    if conf_text:
        summary += f"确认信号：{conf_text}。"
    out.update({
        "leg": leg, "down": down, "state": state, "summary": summary,
        "start": {"date": str(start["date"].date()), "label": start["label"], "price": round(start["price"], 2)},
        "ext": {"date": str(ext_d.date()), "price": round(ext_p, 2), "move": round(move, 4), "elapsed": clock.dist(start["date"], s.index[-1]),
                "elapsed_to_ext": clock.dist(start["date"], ext_d), "unit": "个交易日" if cfg["unit"] == "sessions" else "天"},
        "sub": sub, "cur_sub": cur_sub,
        "turn_level": round(turn_level, 2), "turned": turned,
        "confirm_level": round(conf_level, 2) if conf_text else None, "confirm_text": conf_text, "confirm_what": conf_what, "confirm_date": conf_date if conf_text else None,
        "targets": [t for t in targets], "marks": marks,
        "from_top": round(last_p / top["price"] - 1, 4),
    })
    return out


def levels_table(status: dict, cfg: dict, last_p: float) -> list[dict]:
    """Static and current-leg levels with the side the last close is on."""
    rows = [{"price": float(p), "label": lab, "kind": "static"} for p, lab in cfg["levels"]]
    for p, lab in status.get("targets", []):
        rows.append({"price": float(p), "label": lab, "kind": "target"})
    if status.get("confirm_level"):
        rows.append({"price": status["confirm_level"], "label": f"确认线：{status['confirm_what']}（{status['confirm_date']}）", "kind": "confirm"})
    if status.get("leg") and not status.get("turned"):
        rows.append({"price": status["turn_level"], "label": f"初步信号：从极值{'反弹' if status['down'] else '回落'} {cfg['turn']:.0%}", "kind": "turn"})
    for r in rows:
        r["side"] = "above" if last_p >= r["price"] else "below"
        r["dist"] = round(r["price"] / last_p - 1, 4)
    rows.sort(key=lambda r: -r["price"])
    return rows


# ----------------------------------------------------------------------------------------- sub-waves
SUB_LABELS = {1: {5: ["(i)", "(ii)", "(iii)", "(iv)", "(v)"], 3: ["(a)", "(b)", "(c)", "(d)", "(e)"]},
              2: {5: ["i", "ii", "iii", "iv", "v"], 3: ["a", "b", "c", "d", "e"]}}


def _swing_points(x: np.ndarray, th: float) -> list[int]:
    """Alternating swing points of x (start and end excluded) for a reversal of at least th."""
    piv, trend, lo, hi, ext = [], 0, 0, 0, 0
    for i in range(1, len(x)):
        if trend == 0:
            hi = i if x[i] > x[hi] else hi
            lo = i if x[i] < x[lo] else lo
            if x[i] - x[lo] >= th:
                piv += [lo] if lo else []
                trend, ext = 1, i
            elif x[hi] - x[i] >= th:
                piv += [hi] if hi else []
                trend, ext = -1, i
        elif trend > 0:
            if x[i] >= x[ext]:
                ext = i
            elif x[ext] - x[i] >= th:
                piv.append(ext); trend, ext = -1, i
        else:
            if x[i] <= x[ext]:
                ext = i
            elif x[i] - x[ext] >= th:
                piv.append(ext); trend, ext = 1, i
    return [p for p in piv if 0 < p < len(x) - 1]


def best_split(x: np.ndarray, n_legs: int, min_bars: int = 2, rel: float = 0.08) -> list[int] | None:
    """Split a leg into 5 (motive) or 3 (corrective) sub-waves.

    x is the leg's log price, oriented so the leg rises (x[0] its low, x[-1] its high). Among swing points
    that reverse at least `rel` of the leg, pick the split whose counter-waves are largest while obeying:
    every sub-wave is the extreme of its own span and lasts >= min_bars; for 5 waves, wave 2 stays above
    the start, wave 4 stays above wave 1's end, and wave 3 is not the shortest of 1, 3, 5."""
    n = len(x) - 1
    if n < min_bars * n_legs:
        return None
    pts = _swing_points(x, max(0.004, rel * (x[-1] - x[0])))
    best, best_score = None, -1.0
    if n_legs == 5:
        for i1, i2, i3, i4 in itertools.combinations(pts, 4):
            if min(i1, i2 - i1, i3 - i2, i4 - i3, n - i4) < min_bars:
                continue
            if not (x[i2] > x[0] and x[i4] > x[i1] and x[i3] > x[i1]):
                continue
            if not (x[i1] == x[:i2 + 1].max() and x[i2] == x[i1:i3 + 1].min() and x[i3] == x[i2:i4 + 1].max() and x[i4] == x[i3:].min()):
                continue
            w1, w3, w5 = x[i1] - x[0], x[i3] - x[i2], x[n] - x[i4]
            if w3 < min(w1, w5):
                continue
            score = (x[i1] - x[i2]) + (x[i3] - x[i4])
            if score > best_score:
                best, best_score = [i1, i2, i3, i4], score
    else:
        for i1, i2 in itertools.combinations(pts, 2):
            if min(i1, i2 - i1, n - i2) < min_bars:
                continue
            if x[i2] > x[0] and x[i1] == x[:i2 + 1].max() and x[i2] == x[i1:].min():
                score = x[i1] - x[i2]
                if score > best_score:
                    best, best_score = [i1, i2], score
    return best


def child_kind(n_legs: int, k: int) -> int:
    """Motive sub-waves (1, 3, 5 of an impulse; a and c of a zigzag) have 5 parts, the others 3."""
    return (5 if k % 2 == 0 else 3) if n_legs == 5 else (3 if k % 2 else 5)


def split_node(s: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp, n_legs: int, level: int, max_level: int = 2) -> list[dict]:
    seg = s.loc[d0:d1]
    v = np.log(seg.to_numpy())
    x = v if v[-1] >= v[0] else -v
    cut = None
    for rel, min_bars in ((0.08, 2), (0.04, 2), (0.02, 1)):  # prefer clear swings; accept finer ones only if needed
        cut = best_split(x, n_legs, min_bars, rel)
        if cut is not None:
            break
    if cut is None:
        return []
    ends = [0] + cut + [len(seg) - 1]
    kids = []
    for k in range(n_legs):
        a, b = ends[k], ends[k + 1]
        kind = child_kind(n_legs, k)
        node = {"label": SUB_LABELS[level][n_legs][k], "level": level, "kind": kind,
                "start": [str(seg.index[a].date()), round(float(seg.iloc[a]), 2)], "end": [str(seg.index[b].date()), round(float(seg.iloc[b]), 2)]}
        node["children"] = split_node(s, seg.index[a], seg.index[b], kind, level + 1, max_level) if level < max_level else []
        kids.append(node)
    return kids


def wave_tree(s: pd.Series, cfg: dict, st: dict) -> list[dict]:
    """Every wave of the count (big waves from the config and the tracker) with two degrees of sub-waves."""
    structure = cfg["structure"]
    pts = [(pd.Timestamp(d), lab) for d, lab in cfg["impulse"]] + [(pd.Timestamp(p["date"]), p["label"]) for p in st["pivots"][1:]]
    tree = []
    for (d0, _), (d1, lab) in zip(pts, pts[1:]):
        n_legs = structure[lab]
        tree.append({"label": lab, "level": 0, "kind": n_legs, "start": [str(d0.date()), round(float(s.loc[d0]), 2)], "end": [str(d1.date()), round(float(s.loc[d1]), 2)],
                     "children": split_node(s, d0, d1, n_legs, 1)})
    if st.get("leg") and st["ext"]["date"] != st["start"]["date"] and not st.get("sub"):
        # no internal swing yet: the leg in progress is shown without sub-waves rather than guessing them
        tree.append({"label": st["leg"] + "?", "level": 0, "kind": structure[st["leg"]], "start": [st["start"]["date"], st["start"]["price"]],
                     "end": [st["ext"]["date"], st["ext"]["price"]], "children": []})
    elif st.get("leg") and st["ext"]["date"] != st["start"]["date"]:
        n_legs = structure[st["leg"]]
        d0, d1 = pd.Timestamp(st["start"]["date"]), pd.Timestamp(st["ext"]["date"])
        cuts = [(pd.Timestamp(x["date"]), x["price"]) for x in st.get("sub", [])]
        ends = [(d0, st["start"]["price"])] + cuts + [(d1, st["ext"]["price"])]
        kids = []
        for k, ((a, pa), (b, pb)) in enumerate(zip(ends, ends[1:])):
            last = k == len(ends) - 2
            kind = child_kind(n_legs, k)
            labels = SUB_LABELS[1][n_legs]
            lab = labels[k] if k < len(labels) else f"({k + 1})"
            kids.append({"label": lab + ("?" if last else ""), "level": 1, "kind": kind, "start": [str(a.date()), pa], "end": [str(b.date()), pb],
                         "children": [] if last else split_node(s, a, b, kind, 2)})
        tree.append({"label": st["leg"] + "?", "level": 0, "kind": n_legs, "start": [str(d0.date()), st["start"]["price"]], "end": [str(d1.date()), st["ext"]["price"]], "children": kids})
    return tree

# ----------------------------------------------------------------------------------------- events
def diff_events(asset: str, prev: dict | None, now: dict, final_date: str) -> list[dict]:
    """Compare two settled-close evaluations. prev None = first run: record, don't alert."""
    if prev is None:
        return []
    ev = []

    def add(level: str, text: str):
        ev.append({"asset": asset, "date": final_date, "level": level, "text": text})

    old_piv = {(p["date"], p["label"]) for p in prev.get("pivots", [])}
    for p in now["pivots"]:
        if (p["date"], p["label"]) not in old_piv:
            nxt = f"现在进入 {now['leg']} 浪。" if now["leg"] else "A、B、C 三浪都已走完。"
            add("key", f"{asset} {p['label']} 浪确认结束：终点 {p['price']:,.2f}（{p['date']}），{p.get('confirmed_on', final_date)} 收盘确认。{nxt}")
    if now["state"] != prev.get("state") and now.get("leg") == prev.get("leg"):
        if now["state"] == "turned":
            add("key", f"{asset} " + now["summary"])
        elif now["state"] == "running" and prev.get("state") == "turned":
            add("key", f"{asset} {now['leg']} 浪重新创出{'新低' if now['down'] else '新高'} {now['ext']['price']:,.2f}，之前“可能结束”的信号作废。")
    if now["state"] == "done" and prev.get("state") != "done":
        add("key", f"{asset} " + now["summary"])
    for f in now["flags"]:
        if f not in prev.get("flags", []):
            add("key", f"{asset} {f}")
    if now.get("leg") and now.get("leg") == prev.get("leg") and now["ext"]["date"] != prev.get("ext", {}).get("date") and now["state"] == "running":
        add("info", f"{asset} {now['leg']} 浪{'新低' if now['down'] else '新高'}：收盘 {now['ext']['price']:,.2f}（{now['ext']['date']}）")
    old_side = {(r["label"], r["price"]): r["side"] for r in prev.get("levels", [])}
    for r in now["levels"]:
        o = old_side.get((r["label"], r["price"]))
        if o and o != r["side"] and r["kind"] in ("static", "target", "confirm"):
            add("notice",
                f"{asset} 收盘{'站上' if r['side'] == 'above' else '跌破'} {r['price']:,.2f}（{r['label']}），收于 {now['last'][1]:,.2f}")
    old_marks = {m[0] for m in prev.get("marks", []) if m[0] <= prev["last"][0]}
    for d, lab, _short in now.get("marks", []):
        if d <= final_date and d not in old_marks and d > prev["last"][0]:
            add("info", f"{asset} 到达时间位 {d}：{lab}")
    return ev


def append_events(path: Path, events: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["logged_at", "asset", "date", "level", "text"])
        if new:
            w.writeheader()
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for e in events:
            w.writerow({"logged_at": stamp, **e})


def read_events(path: Path, n: int = 30) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:][::-1]


def email_events(events: list[dict]) -> str:
    sys.path.insert(0, str(ROOT / "code"))
    from email.message import EmailMessage

    from send_signal_email_alerts import load_config, send_email  # the existing SMTP settings and sender
    config = load_config()
    msg = EmailMessage()
    head = events[0]["text"]
    msg["Subject"] = "波浪进度：" + (head if len(head) <= 60 else head[:58] + "…")
    msg["From"], msg["To"] = config["SIGNAL_ALERT_FROM"], config["SIGNAL_ALERT_TO"]
    body = "\n\n".join(f"[{e['date']}] {e['text']}" for e in events)
    body += "\n\n——\n这封邮件由 code/wave_tracker.py 发出，只依据收盘价和 code/waves/wave_count.json 里的数法。波浪数法是主观的，不是交易信号；实盘策略不受影响。"
    msg.set_content(body)
    send_email(msg, config)
    return f"emailed {len(events)} event(s)"


# ----------------------------------------------------------------------------------------- page
def pct(v: float) -> str:
    return f"{v:+.1%}".replace("-", "−")


def tiles(asset: str, st: dict, info: dict) -> list[list[str]]:
    fmt = (lambda v: f"{v:,.2f}") if asset == "QQQ" else (lambda v: f"{v:,.0f}")
    if not st.get("leg"):
        return [["所处位置", "调整已走完", "需要重新数"]]
    e, dn = st["ext"], st["down"]
    t = [["所处位置", f"{st['leg']} 浪" + ("可能已结束" if st["turned"] else "进行中"), (f"内部第 {st['cur_sub']} 段" if st["cur_sub"] and not st["turned"] else f"从 {st['start']['label']} 浪 {fmt(st['start']['price'])} 起")],
         [f"现价（{info.get('label') or ('盘中' if info['provisional'] else '收盘')}）", fmt(info["price"]),
          (f"NQ 期货 {pct(info['proxy']['move'])}，" if info.get("proxy") else "") + f"距 5 浪顶 {pct(st['from_top'])}"],
         [f"本段{'最低' if dn else '最高'}收盘", fmt(e["price"]), f"{e['date']}，本段 {pct(e['move'])}"]]
    if st["confirm_level"]:
        t.append(["确认结束要看", fmt(st["confirm_level"]), "收盘" + ("站上" if dn else "跌破") + f"，距现价 {pct(st['confirm_level'] / info['price'] - 1)}"])
    if not st["turned"]:
        t.append(["初步信号", fmt(st["turn_level"]), f"从极值{'反弹' if dn else '回落'}，距现价 {pct(st['turn_level'] / info['price'] - 1)}"])
    return t


def build_page(out_dir: Path, assets: dict) -> Path:
    hist_path = ROOT / "code" / "waves" / "wave_history.json"
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {}
    raw = {}
    for name, a in assets.items():
        s, st, info, cfg = a["series"], a["live"], a["info"], a["cfg"]
        hist = s.loc[cfg["history_start"]:]
        raw[name] = {"series": [[str(d.date()), round(float(v), 2)] for d, v in hist.items()], "last": [str(s.index[-1].date()), round(float(s.iloc[-1]), 2)],
                     "status": st, "final": a["final"], "info": info, "tiles": tiles(name, st, info), "events": a["events"],
                     "staticSub": cfg.get("static_sub", []), "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     "timing": {"eps": hist.get(name, {}).get("eps", []), "thr": hist.get(name, {}).get("thr", 0.2)}, "histStart": hist.get(name, {}).get("histStart", "")}
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(raw, ensure_ascii=False).replace("</", "<\\/"))
    page = out_dir / "qqq_btc_waves.html"  # page body only, for the private claude.ai artifact
    page.write_text(html, encoding="utf-8")
    # A complete document for a public web host (GitHub Pages): no strategy sell points or private notes,
    # not indexed by search engines, and it reloads itself every 30 minutes while open.
    pub = re.sub(r'<p class="note private">.*?</p>\n?', "", html, flags=re.S)  # strip, not just hide: the source is public too
    pub = re.sub(r'<span class="private">.*?</span>\n?', "", pub, flags=re.S)
    pub = re.sub(r"sold: \[[^\]]*\],", "sold: null,", pub)
    assert "策略卖出" not in pub and "wsquant" not in pub, "private text left in the public page"
    html = pub
    cut = html.index("</style>") + len("</style>")
    full = ('<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta name="robots" content="noindex, nofollow">\n<meta http-equiv="refresh" content="1800">\n'
            + html[:cut] + "\n</head>\n<body>\n<script>window.WAVES_PUBLIC = true;</script>\n" + html[cut:] + "\n</body>\n</html>\n")
    (out_dir / "index.html").write_text(full, encoding="utf-8")
    return page


# ----------------------------------------------------------------------------------------- main
def run_once(out_dir: Path, send: bool, send_all: bool = False) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_all = json.loads(CONFIG.read_text(encoding="utf-8"))
    state_path, events_path = out_dir / "wave_state.json", out_dir / "wave_events.csv"
    prev_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    new_state, all_events, assets = {}, [], {}
    for name in ("QQQ", "BTC"):
        cfg = cfg_all[name]
        s, info = fetch_daily(cfg["ticker"], min(cfg["history_start"], cfg["impulse_start"]))
        if cfg.get("overnight_proxy"):
            s, info = overnight_estimate(s, info, cfg["overnight_proxy"])
        settled = s.iloc[:-1] if info["provisional"] else s
        final = evaluate(settled, cfg)
        final["levels"] = levels_table(final, cfg, float(settled.iloc[-1]))
        final["marks"] = final.get("marks", [])
        live = evaluate(s, cfg)
        live["levels"] = levels_table(live, cfg, float(s.iloc[-1]))
        live["tree"] = wave_tree(s, cfg, live)
        sig = f"{json.dumps(cfg['confirmed'])}|{cfg['current_leg']}"
        prev = prev_state.get(name)
        if prev and prev.get("config_sig") != sig:
            prev = None  # the count was edited by hand: start over without alerting
        ev = diff_events(name, prev, final, str(settled.index[-1].date()))
        final["config_sig"] = sig
        new_state[name] = final
        all_events += ev
        assets[name] = {"series": s, "live": live, "final": {"summary": final["summary"], "last": final["last"], "leg": final.get("leg"), "state": final["state"]},
                        "info": info, "cfg": cfg, "events": []}
    if all_events:
        append_events(events_path, all_events)
    recent = read_events(events_path)
    for name in assets:
        assets[name]["events"] = [e for e in recent if e["asset"] == name][:12]
    state_path.write_text(json.dumps(new_state, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "wave_status.json").write_text(json.dumps({n: {"live": a["live"], "settled": a["final"], "price": a["info"]} for n, a in assets.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    page = build_page(out_dir, assets)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for name, a in assets.items():
        tag = f" (from {a['info']['source']})" if a["info"].get("proxy") else (" (live)" if a["info"]["provisional"] else "")
        print(f"[{stamp}] {name} {a['info']['price']:,.2f}{tag}: {a['live']['summary']}")
    for e in all_events:
        print(f"  EVENT [{e['level']}] {e['text']}")
    key = [e for e in all_events if e["level"] == "key" or (send_all and e["level"] == "notice")]
    if send and key:
        try:
            print("  " + email_events(key))
        except Exception as exc:  # noqa: BLE001 - a mail failure must not stop the tracker
            print(f"  email failed: {exc}", file=sys.stderr)
    print(f"  page: {page}")
    return all_events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", type=int, default=0, help="repeat every N seconds (0 = run once)")
    ap.add_argument("--email", action="store_true", help="email key events (leg possibly over / confirmed over / count invalid) with the signal-alert SMTP settings")
    ap.add_argument("--email-levels", action="store_true", help="with --email, also email every close across a listed price level")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default data/waves)")
    args = ap.parse_args()
    if args.loop and args.loop < 60:
        ap.error("--loop must be at least 60 seconds")
    while True:
        try:
            run_once(args.out, args.email, args.email_levels)
        except Exception as exc:  # noqa: BLE001 - keep looping through network errors
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] update failed: {exc}", file=sys.stderr)
            if not args.loop:
                return 1
        if not args.loop:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
