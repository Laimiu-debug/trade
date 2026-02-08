import datetime as dt
import io
import json
import os
import re
import struct
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _norm_col(s: str) -> str:
    return re.sub(r"[\\s_]+", "", str(s)).lower()


ALIASES = {
    "date": ["date", "日期", "交易日期", "time", "时间"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价"],
    "volume": ["volume", "成交量", "vol", "量"],
    "amount": ["amount", "成交额", "成交金额", "成交额(元)"],
    "code": ["code", "股票代码", "证券代码", "代码"],
    "name": ["name", "股票名称", "名称"],
    "board": ["board", "板块", "市场", "交易板"],
    "industry": ["industry", "行业", "申万行业"],
    "list_days": ["list_days", "上市天数", "上市日数", "上市天数(天)"],
    "list_date": ["list_date", "上市日期", "上市时间"],
    "float_mv": ["float_mv", "流通市值", "流通市值(元)"],
    "market": ["market", "交易所", "市场", "上市市场", "所属市场"],
    "sector_return_5d": ["sector_return_5d", "板块5日涨幅", "行业5日涨幅"],
}

TREND_LABELS = {
    "cond_a": "均线多头",
    "cond_b": "创新高",
    "cond_c": "量价配合",
    "cond_d": "回调健康",
}

BASE_DISPLAY_NAMES = {
    "code": "代码",
    "name": "名称",
    "market_disp": "市场",
    "board": "板块",
    "industry": "行业",
    "close": "收盘价",
    "ideal_buy_price": "理想买点价",
    "buy_distance_pct": "买点距离(%)",
    "score_pct": "综合评分(100分)",
    "trend_conditions": "趋势条件数",
    "trend_met": "趋势满足",
    "trend_missing": "趋势缺失",
    "amount_5d_yi": "5日均成交额(亿)",
    "turn_up": "近期拐头向上",
}


def make_display_names(trend_window: int) -> Dict[str, str]:
    win = int(trend_window)
    names = BASE_DISPLAY_NAMES.copy()
    names.update(
        {
            f"high_{win}d_price": f"{win}日高点价",
            f"return_{win}d_pct": f"{win}日涨幅(%)",
            f"drawdown_{win}_pct": f"当前回撤({win}日%)",
            f"max_drawdown_{win}_pct": f"最大回撤({win}日%)",
            f"drawdown_days_{win}": f"回撤天数({win}日)",
            f"volatility_{win}_pct": f"{win}日波动率(%)",
            f"up_down_ratio_{win}": f"量价比({win}日)",
        }
    )
    return names


def make_display_columns(trend_window: int) -> List[str]:
    win = int(trend_window)
    names = make_display_names(win)
    return [
        names["code"],
        names["name"],
        names["market_disp"],
        names["board"],
        names["industry"],
        names["close"],
        names[f"high_{win}d_price"],
        names["ideal_buy_price"],
        names[f"return_{win}d_pct"],
        names[f"drawdown_{win}_pct"],
        names[f"max_drawdown_{win}_pct"],
        names[f"drawdown_days_{win}"],
        names["buy_distance_pct"],
        names["score_pct"],
        names["trend_conditions"],
        names["trend_met"],
        names["trend_missing"],
        names[f"up_down_ratio_{win}"],
        names["turn_up"],
        names["amount_5d_yi"],
        names[f"volatility_{win}_pct"],
    ]


def make_default_columns(trend_window: int, kind: str = "final") -> List[str]:
    win = int(trend_window)
    names = make_display_names(win)
    base = [
        names["code"],
        names["name"],
        names["market_disp"],
        names["board"],
        names["close"],
        names[f"high_{win}d_price"],
        names["ideal_buy_price"],
        names[f"return_{win}d_pct"],
        names[f"drawdown_{win}_pct"],
        names[f"max_drawdown_{win}_pct"],
        names[f"drawdown_days_{win}"],
        names["buy_distance_pct"],
        names["score_pct"],
        names["trend_conditions"],
        names["trend_met"],
        names[f"up_down_ratio_{win}"],
        names["amount_5d_yi"],
    ]
    if kind == "final":
        base.append(names[f"volatility_{win}_pct"])
    return base


def normalize_display_selection(selected: List[str], trend_window: int, options: List[str]) -> List[str]:
    if not selected:
        return []
    win = int(trend_window)
    mapped = []
    alias_map = {
        "当前回撤(%)": f"当前回撤({win}日%)",
        "最大回撤(20日%)": f"最大回撤({win}日%)",
        "最大回撤(20日)": f"最大回撤({win}日)",
    }
    for label in selected:
        if label in options:
            mapped.append(label)
            continue
        if label in alias_map and alias_map[label] in options:
            mapped.append(alias_map[label])
            continue
        new_label = re.sub(r"\\d+日", f"{win}日", label)
        if new_label in options:
            mapped.append(new_label)
    # de-dup and keep only valid options
    seen = set()
    out = []
    for label in mapped:
        if label in options and label not in seen:
            out.append(label)
            seen.add(label)
    return out

SETTINGS_PATH = Path(__file__).with_name("user_settings.json")


def load_settings() -> Dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_settings(settings: Dict) -> None:
    def _default(o):
        if isinstance(o, (dt.date, dt.datetime)):
            return o.isoformat()
        return str(o)
    SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, default=_default),
        encoding="utf-8",
    )


def init_settings_state() -> None:
    if st.session_state.get("_settings_loaded"):
        return
    settings = load_settings()
    for k, v in settings.items():
        if k in ("eval_date",) and isinstance(v, str):
            try:
                st.session_state[k] = dt.date.fromisoformat(v)
                continue
            except Exception:
                pass
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state["_settings_loaded"] = True


def persist_settings(keys: List[str], sensitive: Optional[List[str]] = None) -> None:
    settings = load_settings()
    sensitive = set(sensitive or [])
    for k in keys:
        if k not in st.session_state:
            continue
        val = st.session_state.get(k)
        if k in sensitive:
            if val:
                settings[k] = val
        else:
            settings[k] = val
    save_settings(settings)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    alias_map = {}
    for k, vs in ALIASES.items():
        for v in vs:
            alias_map[_norm_col(v)] = k
    for col in df.columns:
        key = _norm_col(col)
        if key in alias_map:
            rename_map[col] = alias_map[key]
    return df.rename(columns=rename_map)


def try_read_csv(file_or_path, encoding: str) -> pd.DataFrame:
    if encoding == "自动":
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
            try:
                return pd.read_csv(file_or_path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(file_or_path)
    return pd.read_csv(file_or_path, encoding=encoding)


def load_price_from_file(uploaded_file, encoding: str) -> pd.DataFrame:
    df = try_read_csv(uploaded_file, encoding=encoding)
    return normalize_columns(df)


def load_price_from_folder(folder: Path, encoding: str, code_regex: str) -> pd.DataFrame:
    rows = []
    files = sorted(folder.glob("*.csv"))
    pattern = re.compile(code_regex)
    for f in files:
        df = try_read_csv(f, encoding=encoding)
        df = normalize_columns(df)
        if "code" not in df.columns:
            m = pattern.search(f.stem)
            if m:
                df["code"] = m.group(1)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_meta(uploaded_file, encoding: str) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    df = try_read_csv(uploaded_file, encoding=encoding)
    return normalize_columns(df)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount", "float_mv"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "list_days" in df.columns:
        df["list_days"] = pd.to_numeric(df["list_days"], errors="coerce")
    if "list_date" in df.columns:
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    return df


def normalize_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\\.0$", "", regex=True)
    def _pad(x: str) -> str:
        return x.zfill(6) if x.isdigit() and len(x) <= 6 else x
    return s.map(_pad)


def normalize_market(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "sh": "sh",
        "sz": "sz",
        "bj": "bj",
        "上海": "sh",
        "上证": "sh",
        "深证": "sz",
        "深圳": "sz",
        "北交所": "bj",
        "北京": "bj",
    }
    return s.map(lambda x: mapping.get(x, x))


def infer_market_from_code(code: str) -> str:
    if not isinstance(code, str):
        return ""
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "sz"
    if code.startswith(("43", "83", "87", "88", "92")):
        return "bj"
    return ""


def ensure_market_board(df: pd.DataFrame) -> pd.DataFrame:
    if "market" in df.columns:
        df["market"] = normalize_market(df["market"])
    else:
        df["market"] = df["code"].map(infer_market_from_code)
    if "board" not in df.columns:
        df["board"] = df.apply(lambda r: infer_board(r["code"], r["market"]), axis=1)
    else:
        df["board"] = df["board"].fillna(
            df.apply(lambda r: infer_board(r["code"], r["market"]), axis=1)
        )
    return df


def is_probably_stock_code(code: str, market: str = "") -> bool:
    if not isinstance(code, str):
        return False
    if not code.isdigit() or len(code) != 6:
        return False
    if market == "sh":
        return code.startswith(("600", "601", "603", "605", "688", "689"))
    if market == "sz":
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    if market == "bj":
        return code.startswith(("43", "83", "87", "88", "92"))
    return code.startswith(
        ("600", "601", "603", "605", "688", "689", "000", "001", "002", "003", "300", "301", "43", "83", "87", "88", "92")
    )


def filter_only_stocks(df: pd.DataFrame) -> pd.DataFrame:
    if "market" in df.columns:
        mask = df.apply(lambda r: is_probably_stock_code(str(r["code"]), str(r["market"])), axis=1)
    else:
        mask = df["code"].map(lambda x: is_probably_stock_code(str(x)))
    return df[mask]


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for target, source in mapping.items():
        if source and source in df.columns:
            df[target] = df[source]
    return df


def column_mapping_ui(
    df: pd.DataFrame,
    title: str,
    mapping_targets: Dict[str, str],
    key_prefix: str,
) -> Dict[str, str]:
    if df is None or df.empty:
        return {}
    with st.expander(title, expanded=False):
        st.caption("当表头不规范或不是中文时，在此指定对应列。")
        cols = ["<不使用>"] + list(df.columns)
        mapping = {}
        for target, label in mapping_targets.items():
            default = target if target in df.columns else None
            default_idx = cols.index(default) if default in cols else 0
            sel = st.selectbox(label, cols, index=default_idx, key=f"{key_prefix}_{target}")
            if sel != "<不使用>":
                mapping[target] = sel
        if mapping:
            if len(set(mapping.values())) != len(mapping.values()):
                st.warning("映射列有重复选择，请检查。")
            st.write("当前映射：", mapping)
    return mapping


def parse_headers_json(text: str) -> Dict[str, str]:
    if not text.strip():
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        return {}
    return {}


def fetch_meta_from_api(
    url: str,
    method: str,
    codes: List[str],
    code_param: str,
    headers: Dict[str, str],
    timeout: int,
    data_key: str = "",
    payload_style: str = "comma",
) -> Tuple[pd.DataFrame, str]:
    if not url:
        return pd.DataFrame(), "API地址为空。"
    if not codes:
        return pd.DataFrame(), "代码列表为空。"
    if payload_style == "array":
        code_value = codes
    else:
        code_value = ",".join(codes)

    try:
        if method == "GET":
            params = {code_param: code_value}
            full_url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
            req = urllib.request.Request(full_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
        else:
            body = json.dumps({code_param: code_value}).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
        obj = json.loads(data.decode("utf-8", errors="ignore"))
        if isinstance(obj, dict) and data_key and data_key in obj:
            obj = obj[data_key]
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            obj = obj["data"]
        if isinstance(obj, list):
            df = pd.DataFrame(obj)
            return df, ""
        return pd.DataFrame(), "API返回格式不是列表。"
    except Exception as exc:
        return pd.DataFrame(), f"API调用失败：{exc}"



def detect_file_type(df: pd.DataFrame) -> str:
    cols = {_norm_col(c) for c in df.columns}
    has_price = {"open", "high", "low", "close"}.issubset(cols)
    has_date = "date" in cols
    if has_price and has_date:
        return "price"
    if {"board", "industry", "float_mv", "list_days", "list_date"}.intersection(cols):
        return "meta"
    return "unknown"


def find_tdx_paths(extra_roots: Optional[List[Path]] = None, deep_scan: bool = False, max_depth: int = 4) -> List[Path]:
    roots: List[Path] = []
    for env in ("PROGRAMFILES", "PROGRAMFILES(X86)", "SYSTEMDRIVE"):
        val = os.environ.get(env)
        if val:
            roots.append(Path(val))
    roots += [Path("C:\\"), Path("D:\\"), Path("E:\\")]
    if extra_roots:
        roots += extra_roots

    rels = [
        Path("TdxW_HS") / "vipdoc",
        Path("TdxW") / "vipdoc",
        Path("通达信") / "TdxW_HS" / "vipdoc",
        Path("通达信") / "TdxW" / "vipdoc",
        Path("同花顺") / "TdxW_HS" / "vipdoc",
    ]

    candidates = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for rel in rels:
            p = root / rel
            if p.exists():
                key = str(p).lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(p)
        # one-level search for vendor folder names
        for p in root.glob("*/TdxW_HS/vipdoc"):
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(p)
        for p in root.glob("*/TdxW/vipdoc"):
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(p)

    filtered = [
        p
        for p in candidates
        if (p / "sh" / "lday").exists() or (p / "sz" / "lday").exists()
    ]
    if deep_scan:
        for root in roots:
            if not root.exists():
                continue
            for dirpath, dirnames, _ in os.walk(root):
                depth = len(Path(dirpath).relative_to(root).parts)
                if depth > max_depth:
                    dirnames[:] = []
                    continue
                if Path(dirpath).name.lower() == "vipdoc":
                    p = Path(dirpath)
                    if (p / "sh" / "lday").exists() or (p / "sz" / "lday").exists():
                        if str(p).lower() not in {str(x).lower() for x in filtered}:
                            filtered.append(p)
    return filtered


def resolve_vipdoc_path(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if path.name.lower() == "vipdoc":
        return path
    if (path / "vipdoc").exists():
        return path / "vipdoc"
    for p in path.glob("**/vipdoc"):
        if (p / "sh" / "lday").exists() or (p / "sz" / "lday").exists():
            return p
    return None


def infer_board(code: str, market: str) -> str:
    if market.lower() == "bj":
        return "北交所"
    if code.startswith("300"):
        return "创业板"
    if code.startswith("688") or code.startswith("689"):
        return "科创板"
    if market.lower() in ("sh", "sz"):
        return "主板"
    return "其他"


def read_tdx_day_file(path: Path, max_days: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    record_size = 32
    rows = []
    file_size = path.stat().st_size
    total = file_size // record_size
    if total == 0:
        return pd.DataFrame()
    if max_days is None or max_days <= 0:
        start = 0
        count = total
    else:
        count = min(max_days, total)
        start = total - count
    with path.open("rb") as f:
        f.seek(start * record_size)
        for _ in range(count):
            buf = f.read(record_size)
            if len(buf) != record_size:
                break
            # date, open, high, low, close, amount(float), volume, reserved
            date_i, open_i, high_i, low_i, close_i, amount_f, volume_i, _ = struct.unpack(
                "<IIIIIfII", buf
            )
            rows.append(
                {
                    "date": str(date_i),
                    "open": open_i / 100.0,
                    "high": high_i / 100.0,
                    "low": low_i / 100.0,
                    "close": close_i / 100.0,
                    "amount": float(amount_f),
                    "volume": int(volume_i),
                }
            )
    return pd.DataFrame(rows)


def load_tdx_daily(vipdoc: Path, max_days: int) -> pd.DataFrame:
    rows = []
    markets = {"sh": "sh", "sz": "sz", "bj": "bj"}
    for m in markets.values():
        lday = vipdoc / m / "lday"
        if not lday.exists():
            continue
        for f in lday.glob("*.day"):
            code = re.findall(r"(\\d{6})", f.stem)
            code = code[0] if code else f.stem[-6:]
            df = read_tdx_day_file(f, max_days=max_days)
            if df.empty:
                continue
            df["code"] = code
            df["market"] = m
            df["board"] = infer_board(code, m)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = normalize_columns(df)
    return df


def add_features(df: pd.DataFrame, trend_window: int = 20) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).copy()
    g = df.groupby("code", group_keys=False)

    for win in (5, 10, 20, 60):
        df[f"ma{win}"] = g["close"].rolling(win).mean().reset_index(level=0, drop=True)

    win = max(5, int(trend_window))
    prev_win = max(win * 3, win + 5)
    min_periods = max(5, int(round(win * 0.25)))

    df[f"return_{win}d"] = g["close"].pct_change(win)
    df["return_5d"] = g["close"].pct_change(5)
    df[f"high_{win}d"] = g["high"].rolling(win).max().reset_index(level=0, drop=True)
    df[f"high_{win}d_prev"] = (
        g["high"].shift(win).rolling(prev_win).max().reset_index(level=0, drop=True)
    )
    df[f"drawdown_{win}"] = (df[f"high_{win}d"] - df["close"]) / df[f"high_{win}d"]

    df["is_up"] = df["close"] > df["open"]
    df["is_down"] = df["close"] < df["open"]
    df[f"up_days_{win}"] = (
        g["is_up"].rolling(win, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    df[f"down_days_{win}"] = (
        g["is_down"].rolling(win, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    if "amount" in df.columns:
        df["amt_up"] = np.where(df["is_up"], df["amount"], np.nan)
        df["amt_down"] = np.where(~df["is_up"], df["amount"], np.nan)
        df[f"avg_up_amt_{win}"] = (
            g["amt_up"].rolling(win, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        )
        df[f"avg_down_amt_{win}"] = (
            g["amt_down"].rolling(win, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        )
        df[f"up_down_ratio_{win}"] = df[f"avg_up_amt_{win}"] / (
            df[f"avg_down_amt_{win}"] + 1e-9
        )
    elif "volume" in df.columns:
        df["vol_up"] = np.where(df["is_up"], df["volume"], np.nan)
        df["vol_down"] = np.where(~df["is_up"], df["volume"], np.nan)
        df[f"avg_up_vol_{win}"] = (
            g["vol_up"].rolling(win, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        )
        df[f"avg_down_vol_{win}"] = (
            g["vol_down"].rolling(win, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        )
        df[f"up_down_ratio_{win}"] = df[f"avg_up_vol_{win}"] / (
            df[f"avg_down_vol_{win}"] + 1e-9
        )
    else:
        df[f"up_down_ratio_{win}"] = np.nan

    df["ma5_slope_3"] = (
        g["ma5"].apply(lambda s: (s - s.shift(3)) / s.shift(3)).reset_index(level=0, drop=True)
    )
    df["ma10_slope_5"] = (
        g["ma10"].apply(lambda s: (s - s.shift(5)) / s.shift(5)).reset_index(level=0, drop=True)
    )
    df["ma20_slope_5"] = (
        g["ma20"].apply(lambda s: (s - s.shift(5)) / s.shift(5)).reset_index(level=0, drop=True)
    )

    def _max_drawdown(arr):
        peak = arr[0]
        max_dd = 0.0
        for x in arr[1:]:
            if x > peak:
                peak = x
            if peak > 0:
                dd = (peak - x) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _days_since_high(arr):
        if len(arr) == 0:
            return np.nan
        idx = int(np.nanargmax(arr))
        return len(arr) - 1 - idx

    df[f"max_drawdown_{win}"] = (
        g["close"].rolling(win, min_periods=win).apply(_max_drawdown, raw=True).reset_index(level=0, drop=True)
    )
    df[f"drawdown_days_{win}"] = (
        g["high"].rolling(win, min_periods=min_periods).apply(_days_since_high, raw=True).reset_index(level=0, drop=True)
    )

    df["amplitude"] = (df["high"] - df["low"]) / df["close"]
    df[f"volatility_{win}"] = (
        g["amplitude"].rolling(win).std().reset_index(level=0, drop=True)
    )

    if "amount" in df.columns:
        df["amount_5d"] = g["amount"].rolling(5).mean().reset_index(level=0, drop=True)
    else:
        df["amount_5d"] = np.nan

    return df


def merge_meta(price_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    if meta_df is None or meta_df.empty:
        return price_df
    if "code" not in meta_df.columns:
        return price_df
    merged = price_df.merge(meta_df, on="code", how="left", suffixes=("", "_meta"))
    for col in meta_df.columns:
        if col == "code":
            continue
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[meta_col])
            else:
                merged[col] = merged[meta_col]
            merged = merged.drop(columns=[meta_col])
    return merged


def merge_meta_with_api(meta_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    if api_df is None or api_df.empty:
        return meta_df
    api_df = normalize_columns(api_df)
    if "code" not in api_df.columns:
        return meta_df
    if meta_df is None or meta_df.empty:
        return api_df
    meta_df = meta_df.merge(api_df, on="code", how="left", suffixes=("", "_api"))
    for col in api_df.columns:
        if col == "code":
            continue
        api_col = f"{col}_api"
        if api_col in meta_df.columns:
            if col in meta_df.columns:
                meta_df[col] = meta_df[col].fillna(meta_df[api_col])
            else:
                meta_df[col] = meta_df[api_col]
            meta_df = meta_df.drop(columns=[api_col])
    return meta_df


def compute_scores(df: pd.DataFrame, params: dict, warnings: list) -> pd.DataFrame:
    if df is None:
        return df
    s = df.copy()
    if s.empty:
        s["score"] = np.nan
        return s
    if "sector_return_5d" in s.columns:
        s["sector_score"] = s["sector_return_5d"].rank(pct=True)
    elif "industry" in s.columns:
        sector_ret = s.groupby("industry")["return_5d"].transform("mean")
        s["sector_score"] = sector_ret.rank(pct=True)
    elif "board" in s.columns:
        sector_ret = s.groupby("board")["return_5d"].transform("mean")
        s["sector_score"] = sector_ret.rank(pct=True)
    else:
        s["sector_score"] = 0.5
        warnings.append("缺少板块/行业涨幅字段，题材强度改用中性得分。")

    win = int(params.get("trend_window", 20))
    dd_field = f"drawdown_{win}"
    if params.get("drawdown_mode") == "最大回撤" and f"max_drawdown_{win}" in s.columns:
        dd_field = f"max_drawdown_{win}"
    if dd_field not in s.columns:
        warnings.append("缺少回撤字段，回撤评分将视为0。")
        s["drawdown_score"] = 0.0
    else:
        dd = s[dd_field]
        dd_score = pd.Series(0.0, index=s.index)
        dd_score[(dd >= 0.10) & (dd <= 0.20)] = 1.0
        mid_low = (dd >= 0.05) & (dd < 0.10)
        dd_score[mid_low] = (dd[mid_low] - 0.05) / 0.05
        mid_high = (dd > 0.20) & (dd <= 0.25)
        dd_score[mid_high] = (0.25 - dd[mid_high]) / 0.05
        s["drawdown_score"] = dd_score.fillna(0.0)

    if "amount_5d" in s.columns:
        s["amount_score"] = (s["amount_5d"] >= params["amount_min"]).astype(float)
    else:
        s["amount_score"] = 0.0

    vol_col = f"volatility_{win}"
    if vol_col not in s.columns:
        alt_cols = [c for c in s.columns if c.startswith("volatility_")]
        vol_col = alt_cols[0] if alt_cols else None
    if vol_col:
        vol_rank = s[vol_col].rank(pct=True)
        s["vol_score"] = 1 - vol_rank
    else:
        s["vol_score"] = 0.5

    s["score"] = (
        params["w_sector"] * s["sector_score"].fillna(0)
        + params["w_dd"] * s["drawdown_score"].fillna(0)
        + params["w_amount"] * s["amount_score"].fillna(0)
        + params["w_vol"] * s["vol_score"].fillna(0)
    )
    return s


def apply_layer_filters(
    latest: pd.DataFrame,
    params: dict,
    warnings: list,
) -> dict:
    layers = {}
    win = int(params.get("trend_window", 20))
    ret_col = f"return_{win}d"
    high_col = f"high_{win}d"
    high_prev_col = f"high_{win}d_prev"
    ratio_col = f"up_down_ratio_{win}"
    up_days_col = f"up_days_{win}"
    down_days_col = f"down_days_{win}"
    avg_up_col = f"avg_up_amt_{win}"
    avg_down_col = f"avg_down_amt_{win}"
    if avg_up_col not in latest.columns:
        avg_up_col = f"avg_up_vol_{win}"
        avg_down_col = f"avg_down_vol_{win}"

    # Layer 1: basic filter
    l1 = latest.copy()
    if params.get("enable_market") and "market" in l1.columns:
        l1 = l1[l1["market"].isin(params.get("markets", []))]
    elif params.get("enable_market"):
        warnings.append("缺少市场字段，已跳过市场过滤。")
    if params["enable_board"] and "board" in l1.columns:
        l1 = l1[l1["board"].isin(params["boards"])]
    elif params["enable_board"]:
        warnings.append("缺少板块字段，已跳过板块过滤。")
    if params["enable_st"] and "name" in l1.columns:
        l1 = l1[~l1["name"].astype(str).str.contains("ST", case=False, na=False)]
    elif params["enable_st"]:
        warnings.append("缺少名称字段，已跳过ST过滤。")
    if params["enable_list_days"]:
        if "list_days" in l1.columns:
            l1 = l1[l1["list_days"] > params["min_list_days"]]
        elif "list_date" in l1.columns and params["eval_date"] is not None:
            list_days = (params["eval_date"] - l1["list_date"]).dt.days
            l1 = l1[list_days > params["min_list_days"]]
        else:
            warnings.append("缺少上市天数/日期字段，已跳过上市天数过滤。")
    if params["enable_float_mv"]:
        if "float_mv" in l1.columns:
            l1 = l1[l1["float_mv"] >= params["min_float_mv"]]
        else:
            warnings.append("缺少流通市值字段，已跳过流通市值过滤。")
    layers["layer1"] = l1

    # Layer 2: return filter
    if ret_col not in l1.columns:
        warnings.append("缺少涨幅字段，已跳过涨幅筛选。")
        l2 = l1.copy()
    else:
        l2 = l1[
            (l1[ret_col] >= params["ret_min"])
            & (l1[ret_col] <= params["ret_max"])
        ].copy()
        l2 = l2.sort_values(ret_col, ascending=False).head(params["top_n"])
    layers["layer2"] = l2

    # Layer 3: trend conditions
    cond_a = (l2["ma5"] > l2["ma10"]) & (l2["ma10"] > l2["ma20"]) & (
        l2["ma20"] > l2["ma60"]
    ) & (l2["close"] > l2["ma20"])
    if high_col in l2.columns and high_prev_col in l2.columns:
        cond_b = l2[high_col] > l2[high_prev_col]
    else:
        cond_b = pd.Series(False, index=l2.index)
        warnings.append("缺少创新高所需字段，已跳过创新高条件。")

    min_side_days = max(3, int(round(win * 0.1)))
    if ratio_col in l2.columns:
        cond_c = (
            (l2[ratio_col] > params["up_down_ratio"])
            & (l2.get(up_days_col, 0) >= min_side_days)
            & (l2.get(down_days_col, 0) >= min_side_days)
        )
    elif avg_up_col in l2.columns and avg_down_col in l2.columns:
        cond_c = l2[avg_up_col] > l2[avg_down_col] * params["up_down_ratio"]
    else:
        cond_c = pd.Series(False, index=l2.index)
        warnings.append("缺少量价配合所需字段，已跳过量价配合条件。")

    dd_field = f"drawdown_{win}"
    if params.get("drawdown_mode") == "最大回撤" and f"max_drawdown_{win}" in l2.columns:
        dd_field = f"max_drawdown_{win}"
    cond_d = (l2[dd_field] >= params["dd_min"]) & (
        l2[dd_field] <= params["dd_max"]
    )
    l2["cond_a"] = cond_a.fillna(False)
    l2["cond_b"] = cond_b.fillna(False)
    l2["cond_c"] = cond_c.fillna(False)
    l2["cond_d"] = cond_d.fillna(False)

    cond_e = pd.Series(True, index=l2.index)
    if params.get("enable_turn_up"):
        cond_e = (
            (l2["ma5_slope_3"] >= params.get("turn_up_ma5", 0))
            & (l2["ma10_slope_5"] >= params.get("turn_up_ma10", 0))
            & (l2["ma20_slope_5"] >= params.get("turn_up_ma20", 0))
            & (l2["close"] >= l2["ma10"])
        )
    l2["turn_up"] = cond_e.fillna(False)
    cond_count = (
        cond_a.fillna(False).astype(int)
        + cond_b.fillna(False).astype(int)
        + cond_c.fillna(False).astype(int)
        + cond_d.fillna(False).astype(int)
    )
    l3 = l2[cond_count >= params["trend_min_conditions"]].copy()
    if params.get("enable_turn_up"):
        l3 = l3[l3["turn_up"]]
    l3["trend_conditions"] = cond_count
    layers["layer3"] = l3

    # Layer 4: buy distance
    l3["ideal_buy"] = l3[["ma10", "ma20"]].max(axis=1)
    l3["buy_distance"] = (l3["close"] - l3["ideal_buy"]) / l3["ideal_buy"]
    l4 = l3[
        (l3["buy_distance"] >= params["buy_min"])
        & (l3["buy_distance"] <= params["buy_max"])
    ].copy()
    layers["layer4"] = l4

    # Layer 5: ranking
    l4_scored = compute_scores(l4, params, warnings)
    layers["layer4"] = l4_scored
    if l4_scored is None or l4_scored.empty or "score" not in l4_scored.columns:
        l5 = l4_scored if l4_scored is not None else pd.DataFrame()
    else:
        l5 = l4_scored.sort_values("score", ascending=False).head(params["final_n"])
    layers["layer5"] = l5

    return layers


def format_summary(df: pd.DataFrame, trend_window: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    win = int(trend_window)
    out = df.copy()
    if "market" in out.columns:
        out["market_disp"] = out["market"].map({"sh": "沪市", "sz": "深市", "bj": "北交所"}).fillna(out["market"])
    if "ideal_buy" in out.columns:
        out["ideal_buy_price"] = out["ideal_buy"]
    high_col = f"high_{win}d"
    ret_col = f"return_{win}d"
    dd_col = f"drawdown_{win}"
    max_dd_col = f"max_drawdown_{win}"
    dd_days_col = f"drawdown_days_{win}"
    vol_col = f"volatility_{win}"
    ratio_col = f"up_down_ratio_{win}"

    if high_col in out.columns:
        out[f"high_{win}d_price"] = out[high_col]
    if ret_col in out.columns:
        out[f"return_{win}d_pct"] = out[ret_col] * 100
    if dd_col in out.columns:
        out[f"drawdown_{win}_pct"] = out[dd_col] * 100
    if max_dd_col in out.columns:
        out[f"max_drawdown_{win}_pct"] = out[max_dd_col] * 100
    if dd_days_col in out.columns:
        out[dd_days_col] = out[dd_days_col]
    if "buy_distance" in out.columns:
        out["buy_distance_pct"] = out["buy_distance"] * 100
    if "score" in out.columns:
        out["score_pct"] = out["score"] * 100
    if "amount_5d" in out.columns:
        out["amount_5d_yi"] = out["amount_5d"] / 1e8
    if vol_col in out.columns:
        out[f"volatility_{win}_pct"] = out[vol_col] * 100
    if ratio_col in out.columns:
        out[ratio_col] = out[ratio_col]
    if "turn_up" in out.columns:
        out["turn_up"] = out["turn_up"].map(lambda x: "是" if bool(x) else "否")

    if {"cond_a", "cond_b", "cond_c", "cond_d"}.intersection(out.columns):
        def _trend_met(row):
            return "、".join([label for key, label in TREND_LABELS.items() if bool(row.get(key))])
        def _trend_missing(row):
            return "、".join([label for key, label in TREND_LABELS.items() if not bool(row.get(key))])
        out["trend_met"] = out.apply(_trend_met, axis=1)
        out["trend_missing"] = out.apply(_trend_missing, axis=1)

    cols = [
        "code",
        "name",
        "market_disp",
        "board",
        "industry",
        "close",
        f"high_{win}d_price",
        "ideal_buy_price",
        f"return_{win}d_pct",
        f"drawdown_{win}_pct",
        f"max_drawdown_{win}_pct",
        f"drawdown_days_{win}",
        "buy_distance_pct",
        "score_pct",
        "trend_conditions",
        "trend_met",
        "trend_missing",
        ratio_col,
        "turn_up",
        "amount_5d_yi",
        f"volatility_{win}_pct",
    ]
    keep = [c for c in cols if c in out.columns]
    out = out[keep].copy()
    display_names = make_display_names(win)
    out = out.rename(columns={k: v for k, v in display_names.items() if k in out.columns})
    return out


def style_dataframe(df: pd.DataFrame, up_red: bool = True, trend_window: int = 20):
    def _color(val):
        if pd.isna(val):
            return ""
        if val > 0:
            color = "red" if up_red else "green"
        elif val < 0:
            color = "green" if up_red else "red"
        else:
            color = "black"
        return f"color: {color}"

    styler = df.style
    win = int(trend_window)
    cols_color = [f"{win}日涨幅(%)", "买点距离(%)"]
    for col in cols_color:
        if col in df.columns:
            styler = styler.applymap(_color, subset=[col])

    formatters = {}
    if "收盘价" in df.columns:
        formatters["收盘价"] = "{:.2f}"
    if f"{win}日高点价" in df.columns:
        formatters[f"{win}日高点价"] = "{:.2f}"
    if "理想买点价" in df.columns:
        formatters["理想买点价"] = "{:.2f}"
    if f"{win}日涨幅(%)" in df.columns:
        formatters[f"{win}日涨幅(%)"] = "{:.2f}%"
    if f"当前回撤({win}日%)" in df.columns:
        formatters[f"当前回撤({win}日%)"] = "{:.2f}%"
    if f"最大回撤({win}日%)" in df.columns:
        formatters[f"最大回撤({win}日%)"] = "{:.2f}%"
    if "买点距离(%)" in df.columns:
        formatters["买点距离(%)"] = "{:.2f}%"
    if "综合评分(100分)" in df.columns:
        formatters["综合评分(100分)"] = "{:.2f}"
    if "5日均成交额(亿)" in df.columns:
        formatters["5日均成交额(亿)"] = "{:.2f}"
    if f"{win}日波动率(%)" in df.columns:
        formatters[f"{win}日波动率(%)"] = "{:.2f}%"
    if f"量价比({win}日)" in df.columns:
        formatters[f"量价比({win}日)"] = "{:.2f}"
    if f"回撤天数({win}日)" in df.columns:
        formatters[f"回撤天数({win}日)"] = "{:.0f}"

    if formatters:
        styler = styler.format(formatters)
    return styler


def filter_display_columns(df: pd.DataFrame, selected: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not selected:
        return df
    cols = [c for c in selected if c in df.columns]
    if not cols:
        return df
    return df[cols]


def render_table(
    df: pd.DataFrame,
    use_aggrid: bool,
    height: int = 420,
    up_red: bool = True,
    trend_window: int = 20,
):
    if df is None:
        st.info("暂无数据")
        return
    if use_aggrid:
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
        except Exception:
            st.warning("交互表格不可用，请安装 streamlit-aggrid。")
            st.dataframe(style_dataframe(df, True, trend_window), use_container_width=True, hide_index=True)
            return
        percent_fmt = JsCode(
            "function(params){if(params.value===undefined||params.value===null||params.value==='') return '';"
            "return Number(params.value).toFixed(2)+'%';}"
        )
        num2_fmt = JsCode(
            "function(params){if(params.value===undefined||params.value===null||params.value==='') return '';"
            "return Number(params.value).toFixed(2);}"
        )
        int_fmt = JsCode(
            "function(params){if(params.value===undefined||params.value===null||params.value==='') return '';"
            "return Math.round(Number(params.value));}"
        )
        color_fmt = JsCode(
            "function(params){ if(params.value===undefined||params.value===null) return null;"
            f"var upColor = '{'#e53935' if up_red else '#2ecc71'}';"
            f"var downColor = '{'#2ecc71' if up_red else '#e53935'}';"
            "if(Number(params.value)>0){return {'color': upColor};}"
            "if(Number(params.value)<0){return {'color': downColor};}"
            "return null; }"
        )
        win = int(trend_window)
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            sortable=True,
            filter=True,
            resizable=True,
            editable=False,
            minWidth=110,
            maxWidth=260,
            wrapHeaderText=True,
            autoHeaderHeight=True,
        )
        gb.configure_grid_options(
            suppressMovableColumns=False,
            alwaysShowHorizontalScroll=True,
            domLayout="normal",
            headerHeight=34,
            rowHeight=34,
        )
        if "代码" in df.columns:
            gb.configure_column("代码", minWidth=90, maxWidth=120, pinned="left")
        if "名称" in df.columns:
            gb.configure_column("名称", minWidth=120, maxWidth=180, pinned="left")
        if "市场" in df.columns:
            gb.configure_column("市场", minWidth=80, maxWidth=110)
        if "板块" in df.columns:
            gb.configure_column("板块", minWidth=90, maxWidth=120)
        if "行业" in df.columns:
            gb.configure_column("行业", minWidth=110, maxWidth=160)
        percent_cols = [
            f"{win}日涨幅(%)",
            f"当前回撤({win}日%)",
            f"最大回撤({win}日%)",
            "买点距离(%)",
            f"{win}日波动率(%)",
        ]
        num2_cols = [
            "收盘价",
            f"{win}日高点价",
            "理想买点价",
            "综合评分(100分)",
            "5日均成交额(亿)",
            f"量价比({win}日)",
        ]
        int_cols = [f"回撤天数({win}日)", "趋势条件数"]
        for c in percent_cols:
            if c in df.columns:
                gb.configure_column(c, valueFormatter=percent_fmt, cellStyle=color_fmt)
        for c in num2_cols:
            if c in df.columns:
                gb.configure_column(c, valueFormatter=num2_fmt)
        for c in int_cols:
            if c in df.columns:
                gb.configure_column(c, valueFormatter=int_fmt)
        grid_options = gb.build()
        custom_css = {
            ".ag-header": {"background-color": "#f6f8fa", "border-bottom": "1px solid #e5e7eb"},
            ".ag-header-cell-label": {
                "justify-content": "center",
                "white-space": "normal",
                "line-height": "1.2",
                "font-weight": "600",
            },
            ".ag-header-cell-text": {"white-space": "normal"},
            ".ag-cell": {"font-size": "13px", "line-height": "1.2"},
        }
        AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.NO_UPDATE,
            height=height,
            fit_columns_on_grid_load=False,
            allow_unsafe_jscode=True,
            custom_css=custom_css,
        )
    else:
        st.dataframe(style_dataframe(df, True, trend_window), use_container_width=True, hide_index=True)


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return bio.getvalue()


def df_to_pdf_bytes(df: pd.DataFrame, title: str = "Export") -> Optional[bytes]:
    if df is None or df.empty or len(df.columns) == 0:
        return None
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return None

    font_name = "Helvetica"
    font_paths = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyh.ttf",
        r"C:\\Windows\\Fonts\\simhei.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            try:
                pdfmetrics.registerFont(TTFont("CJK", fp))
                font_name = "CJK"
                break
            except Exception:
                continue

    data = [list(df.columns)]
    for _, row in df.iterrows():
        data.append([str(x) if x is not None else "" for x in row.tolist()])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = font_name
    elements = [Paragraph(title, style), Spacer(1, 8)]
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), font_name, 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(table)
    doc.build(elements)
    return buf.getvalue()


def plot_kline(df: pd.DataFrame, code: str, up_red: bool = True) -> go.Figure:
    d = df[df["code"] == code].sort_values("date").tail(120)
    up_color = "#e53935" if up_red else "#2ecc71"
    down_color = "#2ecc71" if up_red else "#e53935"
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=d["date"],
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="K线",
            increasing_line_color=up_color,
            increasing_fillcolor=up_color,
            decreasing_line_color=down_color,
            decreasing_fillcolor=down_color,
        )
    )
    for win, color in [(5, "#2a9d8f"), (10, "#264653"), (20, "#e76f51"), (60, "#f4a261")]:
        col = f"ma{win}"
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d["date"], y=d[col], name=f"MA{win}", line=dict(color=color)))
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
    )
    return fig


def build_suggestions(layers: dict, params: dict, warnings: List[str], price_df: pd.DataFrame, meta_df: pd.DataFrame) -> List[str]:
    tips = []
    win = int(params.get("trend_window", 20))
    if "name" not in price_df.columns and (meta_df is None or meta_df.empty or "name" not in meta_df.columns):
        tips.append("未发现股票中文名称字段：可上传元数据CSV或用API补充名称。")
    if len(layers["layer2"]) == 0:
        tips.append(f"第2层结果为空：可适当降低{win}日涨幅下限或提高Top N。")
    if len(layers["layer3"]) == 0:
        tips.append("第3层结果为空：可降低“趋势条件最少满足”或放宽回撤范围。")
    if len(layers["layer4"]) == 0:
        tips.append("第4层结果为空：可适当放宽买点距离上下限。")
    if len(layers["layer5"]) == 0:
        tips.append("最终Top为空：检查数据是否完整或放宽筛选条件。")
    if any("成交量" in w for w in warnings):
        tips.append("缺少成交量会影响量价配合判断，建议补齐成交量字段。")
    if params.get("enable_board") and "board" not in price_df.columns:
        tips.append("当前无板块字段，板块过滤将被跳过。可上传元数据补齐板块。")
    return tips


def main():
    st.set_page_config(page_title="趋势票选股系统 v1.0", layout="wide")
    st.title("趋势票选股系统 v1.0")
    st.caption("通达信CSV → 5层漏斗筛选 → 可视化结果")
    init_settings_state()

    with st.sidebar:
        st.subheader("数据输入")
        mode = st.radio(
            "数据来源方式",
            ["通达信本地数据（自动查找）", "CSV 单文件（多股票）", "CSV 文件夹（每股一文件）"],
            help="推荐优先使用通达信本地数据或导出的日线CSV。",
            key="data_mode",
        )
        encoding = st.selectbox("CSV编码", ["自动", "utf-8", "gbk", "gb2312"], key="csv_encoding")

        price_df = None
        if mode == "通达信本地数据（自动查找）":
            st.caption("将自动查找本机通达信 TDX 的 vipdoc 数据目录，并直接读取日线 .day 文件。")
            if "tdx_max_days" in st.session_state:
                max_days = st.number_input(
                    "读取最近 N 个交易日",
                    min_value=80,
                    step=20,
                    help="只读取最近 N 个交易日以加快速度。",
                    key="tdx_max_days",
                )
            else:
                max_days = st.number_input(
                    "读取最近 N 个交易日",
                    min_value=80,
                    value=260,
                    step=20,
                    help="只读取最近 N 个交易日以加快速度。",
                    key="tdx_max_days",
                )
            deep_scan = st.checkbox("深度扫描（慢）", value=False, help="若找不到vipdoc，可开启深度扫描。", key="tdx_deep_scan")
            scan_roots = st.text_input("扫描根目录（可选，多个用逗号）", value="C:\\", key="tdx_scan_roots")
            if st.button("自动查找并载入 TDX 数据"):
                extra_roots = [Path(p.strip()) for p in scan_roots.split(",") if p.strip()]
                paths = find_tdx_paths(extra_roots=extra_roots, deep_scan=deep_scan, max_depth=4)
                st.session_state["tdx_paths"] = [str(p) for p in paths]
            paths = st.session_state.get("tdx_paths", [])
            if paths:
                vipdoc_path = st.selectbox("选择找到的 vipdoc 目录", options=paths, key="tdx_vipdoc_selected")
                if st.button("读取选中的 TDX 数据"):
                    with st.spinner("正在读取 TDX 日线数据..."):
                        st.session_state["tdx_price_df"] = load_tdx_daily(
                            Path(vipdoc_path), max_days=int(max_days)
                        )
            else:
                st.info("未找到 vipdoc，建议检查通达信安装目录或手动导出CSV。")

            manual_vipdoc = st.text_input("手动指定通达信安装目录或 vipdoc 路径（可选）", value="", key="tdx_manual_path")
            if manual_vipdoc and st.button("从手动路径读取"):
                path = resolve_vipdoc_path(Path(manual_vipdoc))
                if path and path.exists():
                    with st.spinner("正在读取 TDX 日线数据..."):
                        st.session_state["tdx_price_df"] = load_tdx_daily(
                            path, max_days=int(max_days)
                        )
                else:
                    st.error("未找到可用的 vipdoc 目录，请检查路径。")

            price_df = st.session_state.get("tdx_price_df")
        elif mode == "CSV 单文件（多股票）":
            file = st.file_uploader("上传行情CSV", type=["csv"])
            if file:
                price_df = load_price_from_file(file, encoding=encoding)
        else:
            folder_path = st.text_input("CSV文件夹路径", value="", key="csv_folder_path")
            code_regex = st.text_input("文件名提取代码正则（需分组）", value=r"(\\d{6})", key="csv_code_regex")
            if folder_path:
                price_df = load_price_from_folder(
                    Path(folder_path), encoding=encoding, code_regex=code_regex
                )

        st.subheader("元数据（可选）")
        st.caption("元数据用于板块/上市天数/流通市值/行业等过滤，不包含K线。")
        meta_file = st.file_uploader("上传元数据CSV", type=["csv"], key="meta")
        meta_df = load_meta(meta_file, encoding=encoding)

        with st.expander("从API获取元数据（可选）", expanded=False):
            api_enable = st.checkbox("启用API接口", key="meta_api_enable")
            api_url = st.text_input("API地址", value="", help="返回JSON列表，包含 code 字段。", key="meta_api_url")
            api_method = st.selectbox("请求方式", ["GET", "POST"], key="meta_api_method")
            api_code_param = st.text_input("代码参数名", value="codes", key="meta_api_code_param")
            api_payload_style = st.selectbox("代码传参方式", ["逗号分隔", "数组"], key="meta_api_payload_style")
            api_data_key = st.text_input("数据字段名（可选）", value="", help="若返回为 {data:[...]}，可填 data。", key="meta_api_data_key")
            api_headers = st.text_area("请求头JSON（可选）", value="", help='如 {"Authorization":"Bearer xxx"}', key="meta_api_headers")
            api_timeout = st.number_input("超时时间(秒)", min_value=3, value=10, step=1, key="meta_api_timeout")
            api_max_codes = st.number_input("最多传入代码数量", min_value=100, value=500, step=100, key="meta_api_max_codes")
            if st.button("调用API获取元数据"):
                if not api_enable:
                    st.error("请先启用API接口。")
                elif price_df is None or price_df.empty:
                    st.error("请先加载行情数据，再调用API。")
                else:
                    codes = sorted(price_df["code"].dropna().unique().tolist())
                    codes = codes[: int(api_max_codes)]
                    headers = parse_headers_json(api_headers)
                    payload_style = "array" if api_payload_style == "数组" else "comma"
                    api_df, err = fetch_meta_from_api(
                        api_url,
                        api_method,
                        codes,
                        api_code_param,
                        headers,
                        int(api_timeout),
                        data_key=api_data_key,
                        payload_style=payload_style,
                    )
                    if err:
                        st.error(err)
                    else:
                        st.success(f"已获取 {len(api_df)} 条元数据。")
                        st.session_state["api_meta_df"] = api_df

        api_meta_df = st.session_state.get("api_meta_df")
        if api_meta_df is not None and not api_meta_df.empty:
            meta_df = merge_meta_with_api(meta_df, api_meta_df)

        price_mapping_targets = {
            "date": "日期列",
            "open": "开盘价列",
            "high": "最高价列",
            "low": "最低价列",
            "close": "收盘价列",
            "volume": "成交量列",
            "amount": "成交额列",
            "code": "股票代码列",
            "name": "股票名称列",
            "market": "市场列（可选）",
            "board": "板块列（可选）",
        }
        meta_mapping_targets = {
            "code": "股票代码列",
            "name": "股票名称列",
            "board": "板块列",
            "industry": "行业列",
            "list_days": "上市天数列",
            "list_date": "上市日期列",
            "float_mv": "流通市值列",
            "sector_return_5d": "板块5日涨幅列",
        }
        price_mapping = column_mapping_ui(price_df, "行情列映射（可编辑表头）", price_mapping_targets, "price_map")
        meta_mapping = column_mapping_ui(meta_df, "元数据列映射（可编辑表头）", meta_mapping_targets, "meta_map")
        price_df = apply_column_mapping(price_df, price_mapping)
        meta_df = apply_column_mapping(meta_df, meta_mapping)

        st.subheader("显示设置")
        color_up_red = st.checkbox("红涨绿跌（A股习惯）", value=True, key="color_up_red")
        use_aggrid = st.checkbox("拖动表头/点击排序（交互表格）", value=True, key="use_aggrid")
        trend_window_preview = int(st.session_state.get("trend_window", 20))
        display_options = make_display_columns(trend_window_preview)
        final_cols_default = make_default_columns(trend_window_preview, "final")
        candidate_cols_default = make_default_columns(trend_window_preview, "candidate")
        # sanitize existing selections after trend window change
        if "final_cols" in st.session_state:
            st.session_state["final_cols"] = normalize_display_selection(
                st.session_state["final_cols"],
                trend_window_preview,
                display_options,
            )
        if "candidate_cols" in st.session_state:
            st.session_state["candidate_cols"] = normalize_display_selection(
                st.session_state["candidate_cols"],
                trend_window_preview,
                display_options,
            )
        final_cols = st.multiselect(
            "最终清单显示列",
            display_options,
            default=normalize_display_selection(
                st.session_state.get("final_cols", final_cols_default),
                trend_window_preview,
                display_options,
            )
            or final_cols_default,
            key="final_cols",
        )
        candidate_cols = st.multiselect(
            "候选池显示列",
            display_options,
            default=normalize_display_selection(
                st.session_state.get("candidate_cols", candidate_cols_default),
                trend_window_preview,
                display_options,
            )
            or candidate_cols_default,
            key="candidate_cols",
        )

        st.subheader("分析参数（改完后点“开始分析”）")
        with st.form("analysis_form"):
            eval_default = dt.date.today()
            if price_df is not None and not price_df.empty and "date" in price_df.columns:
                max_dt = pd.to_datetime(price_df["date"], errors="coerce").max()
                if pd.notna(max_dt):
                    eval_default = max_dt.date()
            eval_date = st.date_input(
                "评估日期（默认最新）",
                value=eval_default,
                key="eval_date",
            )

            st.markdown("**基础过滤**")
            only_stocks = st.checkbox(
                "仅A股股票（排除指数/基金/债券）",
                value=True,
                help="基于代码与市场的简单规则过滤非股票数据。",
                key="only_stocks",
            )
            enable_market = st.checkbox("市场过滤（沪/深/京）", value=True, key="enable_market")
            market_label_map = {"沪市(sh)": "sh", "深市(sz)": "sz", "北交所(bj)": "bj"}
            markets_selected = st.multiselect(
                "交易所",
                list(market_label_map.keys()),
                default=list(market_label_map.keys()),
                key="markets_selected",
            )
            markets = [market_label_map[m] for m in markets_selected]
            enable_board = st.checkbox("板块过滤", value=True, key="enable_board")
            boards = st.multiselect(
                "板块",
                ["主板", "创业板", "科创板", "北交所"],
                default=["主板", "创业板", "科创板", "北交所"],
                key="boards",
            )
            enable_st = st.checkbox("剔除ST", value=True, key="enable_st")
            enable_list_days = st.checkbox("上市天数过滤", value=True, key="enable_list_days")
            min_list_days = st.number_input(
                "最少上市天数",
                min_value=0,
                value=250,
                step=10,
                help="剔除次新股。",
                key="min_list_days",
            )

            enable_float_mv = st.checkbox("流通市值过滤", value=True, key="enable_float_mv")
            float_mv_unit = st.selectbox("流通市值单位", ["亿", "万元", "元"], index=0, key="float_mv_unit")
            float_mv_threshold = st.number_input(
                "最小流通市值",
                value=30.0,
                step=1.0,
                help="仅当有流通市值字段时生效。",
                key="float_mv_threshold",
            )

            st.markdown("**趋势筛选**")
            trend_window = st.number_input(
                "趋势周期(交易日)",
                min_value=5,
                max_value=120,
                value=20,
                step=5,
                help="用于涨幅/回撤/波动率/量价配合等计算。",
                key="trend_window",
            )
            st.caption(
                f"{trend_window}日涨幅=近{trend_window}日涨幅；回撤=距{trend_window}日最高点回落比例；"
                "趋势条件=均线多头/创新高/量价配合/回调健康。"
            )
            ret_min = st.number_input(
                f"{trend_window}日涨幅下限(%)",
                value=20.0,
                step=1.0,
                help=f"例如20表示近{trend_window}日涨幅≥20%。",
                key="ret_min",
            ) / 100
            ret_max = st.number_input(
                f"{trend_window}日涨幅上限(%)",
                value=100.0,
                step=5.0,
                help=f"例如100表示近{trend_window}日涨幅≤100%。",
                key="ret_max",
            ) / 100
            top_n = st.number_input("涨幅Top N", min_value=50, value=300, step=50, key="top_n")
            trend_min_conditions = st.number_input(
                "趋势条件最少满足",
                min_value=1,
                max_value=4,
                value=3,
                help="在均线多头/创新高/量价配合/回调健康中至少满足几条。",
                key="trend_min_conditions",
            )
            up_down_ratio = st.number_input(
                "量价配合阈值",
                value=1.2,
                step=0.1,
                help="上涨日均成交额/下跌日均成交额的最小比值（若缺成交额则用成交量）。",
                key="up_down_ratio",
            )
            dd_min = st.number_input(
                "回撤下限(%)",
                value=5.0,
                step=1.0,
                help=f"回撤=距{trend_window}日最高点的回落比例。",
                key="dd_min",
            ) / 100
            dd_max = st.number_input("回撤上限(%)", value=25.0, step=1.0, key="dd_max") / 100
            if st.session_state.get("drawdown_mode") not in ("当前回撤", "最大回撤"):
                if str(st.session_state.get("drawdown_mode", "")).startswith("最大回撤"):
                    st.session_state["drawdown_mode"] = "最大回撤"
                else:
                    st.session_state["drawdown_mode"] = "当前回撤"
            drawdown_mode = st.selectbox(
                "回撤口径",
                ["当前回撤", "最大回撤"],
                index=0,
                format_func=lambda x: x if x == "当前回撤" else f"最大回撤({trend_window}日)",
                help="用于趋势条件与评分的回撤口径。",
                key="drawdown_mode",
            )
            st.markdown("**近期拐头向上**")
            enable_turn_up = st.checkbox(
                "启用近期拐头向上过滤",
                value=False,
                help="要求短期均线斜率转正，避免买在上涨波段尾部。",
                key="enable_turn_up",
            )
            turn_up_ma5 = st.number_input(
                "MA5 3日斜率下限(%)",
                value=0.0,
                step=0.1,
                key="turn_up_ma5",
            ) / 100
            turn_up_ma10 = st.number_input(
                "MA10 5日斜率下限(%)",
                value=0.0,
                step=0.1,
                key="turn_up_ma10",
            ) / 100
            turn_up_ma20 = st.number_input(
                "MA20 5日斜率下限(%)",
                value=0.0,
                step=0.1,
                key="turn_up_ma20",
            ) / 100

            st.markdown("**买点距离**")
            st.caption("理想买点=MAX(MA10, MA20)。买点距离= (收盘价-理想买点)/理想买点。")
            buy_min = st.number_input(
                "买点距离下限(%)",
                value=-5.0,
                step=1.0,
                help="买点距离= (收盘价-理想买点)/理想买点。负值表示低于买点。",
                key="buy_min",
            ) / 100
            buy_max = st.number_input(
                "买点距离上限(%)",
                value=10.0,
                step=1.0,
                help="正值表示高于理想买点。",
                key="buy_max",
            ) / 100

            st.markdown("**排序权重**")
            st.caption("综合评分=题材强度+回撤适中+成交额+波动率，各权重之和不必等于1。")
            w_sector = st.slider("题材强度", 0.0, 1.0, 0.4, 0.05, key="w_sector")
            w_dd = st.slider("回撤适中", 0.0, 1.0, 0.25, 0.05, key="w_dd")
            w_amount = st.slider("成交额", 0.0, 1.0, 0.2, 0.05, key="w_amount")
            w_vol = st.slider("波动率", 0.0, 1.0, 0.15, 0.05, key="w_vol")
            amount_min_yi = st.number_input(
                "日均成交额阈值(亿)",
                value=5.0,
                step=1.0,
                help="近5日均成交额高于阈值则得满分。",
                key="amount_min_yi",
            )
            final_n = st.number_input("最终保留数量", min_value=1, max_value=20, value=5, key="final_n")

            submit_analysis = st.form_submit_button("开始分析")

        if submit_analysis:
            st.session_state["analysis_triggered"] = True

        # persist settings after sidebar render
        persist_settings(
            keys=[
                "data_mode",
                "csv_encoding",
                "tdx_max_days",
                "tdx_deep_scan",
                "tdx_scan_roots",
                "tdx_vipdoc_selected",
                "tdx_manual_path",
                "csv_folder_path",
                "csv_code_regex",
                "meta_api_enable",
                "meta_api_url",
                "meta_api_method",
                "meta_api_code_param",
                "meta_api_payload_style",
                "meta_api_data_key",
                "meta_api_headers",
                "meta_api_timeout",
                "meta_api_max_codes",
                "color_up_red",
                "use_aggrid",
                "final_cols",
                "candidate_cols",
                "eval_date",
                "only_stocks",
                "enable_market",
                "markets_selected",
                "enable_board",
                "boards",
                "enable_st",
                "enable_list_days",
                "min_list_days",
                "enable_float_mv",
                "float_mv_unit",
                "float_mv_threshold",
                "ret_min",
                "ret_max",
                "top_n",
                "trend_window",
                "trend_min_conditions",
                "up_down_ratio",
                "dd_min",
                "dd_max",
                "drawdown_mode",
                "enable_turn_up",
                "turn_up_ma5",
                "turn_up_ma10",
                "turn_up_ma20",
                "buy_min",
                "buy_max",
                "w_sector",
                "w_dd",
                "w_amount",
                "w_vol",
                "amount_min_yi",
                "final_n",
            ],
            sensitive=[],
        )

    if price_df is None or price_df.empty:
        st.info("请先加载行情数据（CSV 或 TDX 本地数据）。")
        st.stop()

    price_df = coerce_types(price_df)
    meta_df = coerce_types(meta_df)
    if meta_df is not None and not meta_df.empty and "code" in meta_df.columns:
        meta_df["code"] = normalize_code(meta_df["code"])
    price_df = merge_meta(price_df, meta_df)

    # file type sanity check
    if meta_df is not None and not meta_df.empty:
        if detect_file_type(meta_df) == "price":
            st.warning("你上传的“元数据CSV”看起来像行情数据，请确认是否选错。")
    if detect_file_type(price_df) != "price":
        st.warning("行情数据缺少开高低收/日期字段，可能选错文件。")

    if "code" not in price_df.columns:
        st.error("行情数据缺少股票代码字段，请在CSV中加入“代码/股票代码”。")
        st.stop()
    price_df["code"] = normalize_code(price_df["code"])
    missing_code = price_df["code"].isna().sum()
    if missing_code > 0:
        st.warning(f"存在 {missing_code} 行缺少代码，已自动剔除。")
        price_df = price_df[price_df["code"].notna()]
    if "date" not in price_df.columns:
        st.error("行情数据缺少日期字段，请在CSV中加入“日期/交易日期”。")
        st.stop()

    price_df = ensure_market_board(price_df)
    if only_stocks:
        before_cnt = len(price_df)
        price_df = filter_only_stocks(price_df)
        filtered_cnt = before_cnt - len(price_df)
        if filtered_cnt > 0:
            st.info(f"已过滤 {filtered_cnt} 行非股票数据。")

    price_type = detect_file_type(price_df)
    meta_type = detect_file_type(meta_df) if meta_df is not None and not meta_df.empty else "未提供"

    st.subheader("数据诊断")
    st.caption(f"行情数据识别结果：{price_type}；元数据识别结果：{meta_type}")
    st.write(
        f"股票数：{price_df['code'].nunique()}，数据行数：{len(price_df):,}"
    )

    # analysis gating and caching
    signature = (
        int(price_df["code"].nunique()),
        int(len(price_df)),
        str(pd.to_datetime(price_df["date"], errors="coerce").max()) if "date" in price_df.columns else "",
    )
    if submit_analysis:
        st.session_state["analysis_signature"] = signature
    if "analysis_signature" in st.session_state and st.session_state["analysis_signature"] != signature:
        st.info("数据已变化，请重新点击“开始分析”。")
        st.session_state.pop("analysis_cache", None)
        st.session_state["analysis_triggered"] = False
        st.stop()
    if not st.session_state.get("analysis_triggered"):
        st.info("请在左侧设置参数后点击“开始分析”。")
        st.stop()

    if submit_analysis or "analysis_cache" not in st.session_state:
        df = price_df.copy()
        if eval_date is not None:
            df = df[df["date"] <= pd.to_datetime(eval_date)]

        warnings = []
        if "amount" not in df.columns and "volume" in df.columns:
            df["amount"] = df["close"] * df["volume"]
            warnings.append("缺少成交额字段，已使用收盘价×成交量估算。")
        if "volume" not in df.columns and "amount" not in df.columns:
            warnings.append("缺少成交量/成交额字段，量价配合条件可能无法满足。")
        elif "volume" not in df.columns:
            warnings.append("缺少成交量字段，量价配合将改用成交额。")

        df = add_features(df, trend_window=int(trend_window))
        latest = df.groupby("code").tail(1).copy()

        float_mv_threshold_value = float_mv_threshold
        if float_mv_unit == "亿":
            float_mv_threshold_value = float_mv_threshold * 1e8
        elif float_mv_unit == "万元":
            float_mv_threshold_value = float_mv_threshold * 1e4

        params = {
            "enable_market": enable_market,
            "markets": markets,
            "enable_board": enable_board,
            "boards": boards,
            "enable_st": enable_st,
            "enable_list_days": enable_list_days,
            "min_list_days": min_list_days,
            "enable_float_mv": enable_float_mv,
            "min_float_mv": float_mv_threshold_value,
            "ret_min": ret_min,
            "ret_max": ret_max,
            "top_n": int(top_n),
            "trend_window": int(trend_window),
            "trend_min_conditions": int(trend_min_conditions),
            "up_down_ratio": up_down_ratio,
            "dd_min": dd_min,
            "dd_max": dd_max,
            "drawdown_mode": drawdown_mode,
            "enable_turn_up": enable_turn_up,
            "turn_up_ma5": turn_up_ma5,
            "turn_up_ma10": turn_up_ma10,
            "turn_up_ma20": turn_up_ma20,
            "buy_min": buy_min,
            "buy_max": buy_max,
            "w_sector": w_sector,
            "w_dd": w_dd,
            "w_amount": w_amount,
            "w_vol": w_vol,
            "amount_min": amount_min_yi * 1e8,
            "final_n": int(final_n),
            "eval_date": pd.to_datetime(eval_date) if eval_date else None,
        }

        layers = apply_layer_filters(latest, params, warnings)
        st.session_state["analysis_cache"] = {
            "df": df,
            "latest": latest,
            "layers": layers,
            "warnings": warnings,
            "params": params,
        }
    cache = st.session_state.get("analysis_cache", {})
    df = cache.get("df")
    latest = cache.get("latest")
    layers = cache.get("layers")
    warnings = cache.get("warnings", [])
    params = cache.get("params", {})

    st.subheader("筛选结果概览")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("第1层", len(layers["layer1"]))
    col2.metric("第2层", len(layers["layer2"]))
    col3.metric("第3层", len(layers["layer3"]))
    col4.metric("第4层", len(layers["layer4"]))
    col5.metric("最终Top", len(layers["layer5"]))

    if warnings:
        st.warning("\n".join(warnings))

    st.subheader("操作与分析建议")
    tips = build_suggestions(layers, params, warnings, price_df, meta_df)
    if tips:
        st.info("\n".join([f"- {t}" for t in tips]))
    else:
        st.info("当前参数下结果正常，可继续观察筛选结果。")

    st.subheader("最终待买清单")
    st.caption("说明：趋势条件=均线多头/创新高/量价配合/回调健康；趋势满足/缺失展示具体条件。")
    final_df = format_summary(layers["layer5"], int(params.get("trend_window", 20)))
    final_df = filter_display_columns(final_df, final_cols)
    render_table(
        final_df,
        use_aggrid=use_aggrid,
        height=420,
        up_red=color_up_red,
        trend_window=int(params.get("trend_window", 20)),
    )
    st.markdown("**导出**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "CSV",
            data=final_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="trend_final_list.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        try:
            excel_bytes = df_to_excel_bytes(final_df)
            st.download_button(
                "Excel",
                data=excel_bytes,
                file_name="trend_final_list.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            st.caption("Excel 需安装 openpyxl")
    with c3:
        pdf_bytes = df_to_pdf_bytes(final_df, title="Final List")
        if pdf_bytes:
            st.download_button(
                "PDF",
                data=pdf_bytes,
                file_name="trend_final_list.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("PDF 需安装 reportlab")

    st.subheader("候选池")
    candidate_df = format_summary(layers["layer4"], int(params.get("trend_window", 20)))
    candidate_df = filter_display_columns(candidate_df, candidate_cols)
    render_table(
        candidate_df,
        use_aggrid=use_aggrid,
        height=360,
        up_red=color_up_red,
        trend_window=int(params.get("trend_window", 20)),
    )
    st.markdown("**导出**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "CSV",
            data=candidate_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="trend_candidate_list.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        try:
            cand_excel = df_to_excel_bytes(candidate_df)
            st.download_button(
                "Excel",
                data=cand_excel,
                file_name="trend_candidate_list.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            st.caption("Excel 需安装 openpyxl")
    with c3:
        cand_pdf = df_to_pdf_bytes(candidate_df, title="Candidate List")
        if cand_pdf:
            st.download_button(
                "PDF",
                data=cand_pdf,
                file_name="trend_candidate_list.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("PDF 需安装 reportlab")

    st.subheader("个股走势")
    options = layers["layer5"]["code"].tolist() or layers["layer4"]["code"].tolist()
    if options:
        name_map = {}
        if "name" in price_df.columns:
            name_map = (
                price_df.dropna(subset=["name"])
                .drop_duplicates("code")
                .set_index("code")["name"]
                .to_dict()
            )
        code = st.selectbox("选择股票", options, format_func=lambda c: f"{c} {name_map.get(c, '')}".strip())
        fig = plot_kline(df, code, up_red=color_up_red)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无可视化股票，请调整筛选参数。")


if __name__ == "__main__":
    main()

