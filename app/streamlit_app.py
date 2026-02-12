import datetime as dt
import io
import json
import os
import re
import struct
import urllib.parse
import urllib.request
import zipfile
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
    "structure_hhh": "结构(HH/HL/HC)",
    "structure_up_score": "上升结构分(0-3)",
    "wyckoff_phase": "威科夫阶段",
    "wyckoff_signal": "关键信号",
    "wy_events_present": "事件覆盖(回看)",
    "wy_sequence_ok": "序列完整(8步)",
}

WYCKOFF_PHASE_HINTS = {
    "A阶段-止跌初期": "观察PS/SC/AR是否连续出现，避免盲目追高。",
    "B阶段-吸筹震荡": "区间震荡为主，关注ST及后续Spring/SOS确认。",
    "C阶段-Spring测试": "关注假跌破后快速收回，若伴随TSO更强。",
    "D阶段-上破准备": "需求主导增强，JOC后优先等待回踩确认。",
    "E阶段-拉升(Markup)": "趋势上行阶段，重视仓位与回撤管理。",
    "A阶段-见顶初期": "上涨秩序开始破坏，谨慎追涨。",
    "B阶段-派发震荡": "高位区间反复，关注冲高回落风险。",
    "C阶段-UTAD": "假突破后回落，警惕派发完成转弱。",
    "D阶段-下破准备": "供给主导增强，反弹偏减仓而非追买。",
    "E阶段-下跌(Markdown)": "趋势下行阶段，以防守为主。",
    "阶段未明": "结构信号不足，建议继续观察。",
}

WYCKOFF_EVENT_OPTIONS = ["PS", "SC", "AR", "ST", "TSO", "Spring", "SOS", "JOC", "LPS", "UTAD", "SOW", "LPSY"]
WYCKOFF_EVENT_LABELS = {
    "PS": "PS (Preliminary Support 初步支撑)",
    "SC": "SC (Selling Climax 卖出高潮)",
    "AR": "AR (Automatic Rally 自动反弹)",
    "ST": "ST (Secondary Test 二次测试)",
    "TSO": "TSO (Terminal Shakeout 终极震仓)",
    "Spring": "Spring (弹簧/假跌破)",
    "SOS": "SOS (Sign of Strength 强势信号)",
    "JOC": "JOC (Jump Over Creek 跃过小溪)",
    "LPS": "LPS (Last Point of Support 最后支撑点)",
    "UTAD": "UTAD (Upthrust After Distribution 派发后假突破)",
    "SOW": "SOW (Sign of Weakness 弱势信号)",
    "LPSY": "LPSY (Last Point of Supply 最后供应点)",
}
WYCKOFF_EVENT_PREFIX = {
    "PS": "ps",
    "SC": "sc",
    "AR": "ar",
    "ST": "st",
    "TSO": "tso",
    "Spring": "spring",
    "SOS": "sos",
    "JOC": "joc",
    "LPS": "lps",
    "UTAD": "utad",
    "SOW": "sow",
    "LPSY": "lpsy",
}
WYCKOFF_PHASE_OPTIONS = list(WYCKOFF_PHASE_HINTS.keys())


def format_wyckoff_event_label(evt: str) -> str:
    return WYCKOFF_EVENT_LABELS.get(str(evt), str(evt))

BACKTEST_EXIT_REASON_LABELS = {
    "stop_loss": "止损(stop_loss)",
    "take_profit": "止盈(take_profit)",
    "event_exit": "事件离场(event_exit)",
    "time_exit": "超时离场(time_exit)",
    "eod_exit": "样本末离场(eod_exit)",
}

BACKTEST_ENTRY_EVENT_WEIGHTS = {
    "PS": 1.0,
    "SC": 1.2,
    "AR": 1.4,
    "ST": 1.6,
    "TSO": 2.5,
    "Spring": 3.0,
    "SOS": 3.4,
    "JOC": 4.0,
    "LPS": 2.8,
    "UTAD": 1.5,
    "SOW": 1.5,
    "LPSY": 1.3,
}

BACKTEST_PRIORITY_MODE_LABELS = {
    "phase_first": "阶段优先(早段性价比)",
    "balanced": "均衡(默认)",
    "momentum": "动量优先(强势延续)",
}

BACKTEST_TRADE_COL_LABELS = {
    "code": "代码(code)",
    "signal_date": "信号日(signal_date)",
    "entry_date": "入场日(entry_date)",
    "exit_date": "离场日(exit_date)",
    "entry_signal": "入场信号(entry_signal)",
    "entry_quality_score": "入场优先分(entry_quality_score)",
    "entry_phase": "阶段(entry_phase)",
    "entry_phase_score": "阶段分(entry_phase_score)",
    "entry_events_weight": "事件强度分(entry_events_weight)",
    "entry_structure_score": "结构分(entry_structure_score)",
    "entry_trend_score": "趋势分(entry_trend_score)",
    "entry_volatility_score": "波动分(entry_volatility_score)",
    "entry_price": "入场价(entry_price)",
    "exit_price": "离场价(exit_price)",
    "bars_held": "持有K线数(bars_held)",
    "exit_reason": "离场原因(exit_reason)",
    "shares": "股数(shares)",
    "position_value": "开仓金额(position_value)",
    "exit_value": "平仓金额(exit_value)",
    "pnl_amount": "盈亏金额(pnl_amount)",
    "ret_pct": "收益率(ret_pct, %)",
}

def make_display_names(trend_window: int) -> Dict[str, str]:
    win = int(trend_window)
    names = BASE_DISPLAY_NAMES.copy()
    names.update(
        {
            f"high_{win}d_price": f"{win}鏃ラ珮鐐逛环",
            f"return_{win}d_pct": f"{win}鏃ユ定骞?%)",
            f"drawdown_{win}_pct": f"褰撳墠鍥炴挙({win}鏃?)",
            f"max_drawdown_{win}_pct": f"鏈€澶у洖鎾?{win}鏃?)",
            f"drawdown_days_{win}": f"鍥炴挙澶╂暟({win}鏃?",
            f"volatility_{win}_pct": f"{win}鏃ユ尝鍔ㄧ巼(%)",
            f"up_down_ratio_{win}": f"閲忎环姣?{win}鏃?",
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
        names["structure_hhh"],
        names["structure_up_score"],
        names["wyckoff_phase"],
        names["wyckoff_signal"],
        names["wy_events_present"],
        names["wy_sequence_ok"],
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
        names["structure_hhh"],
        names["structure_up_score"],
        names["wyckoff_phase"],
        names["wyckoff_signal"],
        names["wy_events_present"],
        names["wy_sequence_ok"],
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
        if k in ("eval_date", "bt_start_date", "bt_end_date") and isinstance(v, str):
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


def sanitize_widget_state() -> None:
    def _normalize_scalar(value: str, aliases: Dict[str, str]) -> str:
        s = str(value).strip()
        if s in aliases:
            return aliases[s]
        sl = s.lower()
        if sl in aliases:
            return aliases[sl]
        return s

    def _sanitize_choice(key: str, options: List[str], aliases: Optional[Dict[str, str]] = None) -> None:
        if key not in st.session_state:
            return
        val = st.session_state.get(key)
        if not isinstance(val, str):
            st.session_state.pop(key, None)
            return
        aliases = aliases or {}
        norm = _normalize_scalar(val, aliases)
        if norm in options:
            st.session_state[key] = norm
            return
        # Fuzzy fallback for severe mojibake / old labels.
        if key == "data_mode":
            if ("通达信" in norm) or ("TDX" in norm) or ("閫氳揪" in norm):
                st.session_state[key] = "TDX本地数据"
                return
            if ("CSV单文件" in norm) or ("鍗曟枃浠" in norm):
                st.session_state[key] = "CSV单文件(多股票)"
                return
            if ("CSV文件夹" in norm) or ("鏂囦欢澶" in norm):
                st.session_state[key] = "CSV文件夹(每股一文件)"
                return
        st.session_state.pop(key, None)

    def _sanitize_multi(
        key: str,
        options: List[str],
        aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        if key not in st.session_state:
            return
        raw = st.session_state.get(key)
        if isinstance(raw, str):
            raw_list = [raw]
        elif isinstance(raw, (list, tuple, set)):
            raw_list = list(raw)
        else:
            st.session_state.pop(key, None)
            return

        aliases = aliases or {}
        cleaned = []
        for item in raw_list:
            norm = _normalize_scalar(str(item), aliases)
            if norm in options and norm not in cleaned:
                cleaned.append(norm)
        if cleaned:
            st.session_state[key] = cleaned
        else:
            st.session_state.pop(key, None)

    _sanitize_choice(
        "data_mode",
        ["TDX本地数据", "CSV单文件(多股票)", "CSV文件夹(每股一文件)"],
        aliases={
            "通达信本地数据（自动查找）": "TDX本地数据",
            "通达信本地数据(自动查找)": "TDX本地数据",
            "tdx本地数据": "TDX本地数据",
            "csv单文件（多股票）": "CSV单文件(多股票)",
            "csv文件夹（每股一文件）": "CSV文件夹(每股一文件)",
        },
    )
    _sanitize_choice(
        "csv_encoding",
        ["auto", "utf-8", "gbk", "gb2312"],
        aliases={
            "自动": "auto",
            "鑷姩": "auto",
            "utf8": "utf-8",
            "utf-8-sig": "utf-8",
        },
    )
    _sanitize_choice("meta_api_method", ["GET", "POST"], aliases={"get": "GET", "post": "POST"})
    _sanitize_choice(
        "meta_api_payload_style",
        ["comma", "array"],
        aliases={"逗号分隔": "comma", "数组": "array", "閫楀彿鍒嗛殧": "comma"},
    )
    _sanitize_choice(
        "float_mv_unit",
        ["亿", "万元", "元"],
        aliases={"億元": "亿", "万元 ": "万元", "浜": "亿"},
    )
    _sanitize_choice(
        "drawdown_mode",
        ["当前回撤", "最大回撤"],
        aliases={"鏈€澶у洖鎾": "最大回撤", "当前回撤(%)": "当前回撤"},
    )
    _sanitize_choice("bt_pool", ["All symbols", "Layer1", "Layer4 candidates", "Final Top", "Wyckoff Phase Pool"])
    _sanitize_choice(
        "bt_range_mode",
        ["lookback_bars", "custom_dates"],
        aliases={
            "按最近k线数": "lookback_bars",
            "按最近K线数": "lookback_bars",
            "自定义日期区间": "custom_dates",
            "lookback": "lookback_bars",
            "custom": "custom_dates",
        },
    )
    _sanitize_choice("wy_phase_scope", ["All symbols", "Layer1", "Layer4 candidates", "Final Top"])
    _sanitize_choice(
        "bt_priority_mode",
        ["phase_first", "balanced", "momentum"],
        aliases={
            "阶段优先": "phase_first",
            "均衡": "balanced",
            "动量优先": "momentum",
        },
    )
    _sanitize_choice(
        "bt_position_mode",
        ["min", "fixed", "risk"],
        aliases={
            "取最小(固定∩风险)": "min",
            "固定仓位": "fixed",
            "风险仓位": "risk",
        },
    )

    _sanitize_multi(
        "markets_selected",
        ["沪市(sh)", "深市(sz)", "北交所(bj)"],
        aliases={
            "沪市": "沪市(sh)",
            "深市": "深市(sz)",
            "北交所": "北交所(bj)",
            "娌競(sh)": "沪市(sh)",
            "娣卞競(sz)": "深市(sz)",
            "鍖椾氦鎵€(bj)": "北交所(bj)",
        },
    )
    _sanitize_multi(
        "boards",
        ["主板", "创业板", "科创板", "北交所"],
        aliases={
            "涓绘澘": "主板",
            "鍒涗笟鏉": "创业板",
            "绉戝垱鏉": "科创板",
            "鍖椾氦鎵€": "北交所",
        },
    )
    wy_events = WYCKOFF_EVENT_OPTIONS
    _sanitize_multi("wy_required_events", wy_events)
    _sanitize_multi("bt_entry_events", wy_events)
    _sanitize_multi("bt_exit_events", wy_events)
    _sanitize_multi("wy_phase_events", wy_events)
    _sanitize_multi("wy_phase_selected", WYCKOFF_PHASE_OPTIONS)


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
    if encoding in ("自动", "auto", "AUTO", "鑷姩"):
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
    if not folder.exists():
        raise FileNotFoundError(f"CSV文件夹不存在：{folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"路径不是文件夹：{folder}")
    try:
        pattern = re.compile(code_regex)
    except re.error as e:
        raise ValueError(f"文件名提取代码正则无效：{e}") from e

    rows = []
    files = sorted(folder.glob("*.csv"))
    for f in files:
        try:
            df = try_read_csv(f, encoding=encoding)
        except Exception:
            continue
        df = normalize_columns(df)
        if "code" not in df.columns:
            m = pattern.search(f.stem)
            if m:
                df["code"] = m.group(1)
            else:
                continue
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
        "沪市": "sh",
        "深圳": "sz",
        "深证": "sz",
        "深市": "sz",
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
        st.caption("当表头不规范时，可以在这里手动映射列。")
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
                st.warning("映射列有重复，请检查。")
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


def dedupe_paths(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


@st.cache_data(show_spinner=False, ttl=1800)
def cached_tdx_paths(deep_scan: bool = False) -> List[str]:
    return [str(p) for p in find_tdx_paths(deep_scan=deep_scan)]


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

    # Wyckoff-friendly micro structure: higher high / higher low / higher close
    df["prev_high"] = g["high"].shift(1)
    df["prev_low"] = g["low"].shift(1)
    df["prev_close"] = g["close"].shift(1)
    df["hh"] = df["high"] > df["prev_high"]
    df["hl"] = df["low"] > df["prev_low"]
    df["hc"] = df["close"] > df["prev_close"]
    df["lh"] = df["high"] < df["prev_high"]
    df["ll"] = df["low"] < df["prev_low"]
    df["lc"] = df["close"] < df["prev_close"]

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
    df["ma20_slope_10"] = (
        g["ma20"].apply(lambda s: (s - s.shift(10)) / s.shift(10)).reset_index(level=0, drop=True)
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
    df["bar_range"] = df["high"] - df["low"]
    df["close_pos"] = (df["close"] - df["low"]) / (df["bar_range"] + 1e-9)
    df["bar_range_pct"] = df["bar_range"] / (df["prev_close"].abs() + 1e-9)
    df["bar_range_pct_ma20"] = (
        g["bar_range_pct"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    )

    if "amount" in df.columns:
        df["amount_5d"] = g["amount"].rolling(5).mean().reset_index(level=0, drop=True)
    else:
        df["amount_5d"] = np.nan

    # Wyckoff structure context with a 60-day trading range
    wy_win = 60
    wy_min_periods = max(20, int(round(wy_win * 0.35)))
    df[f"tr_high_{wy_win}"] = (
        g["high"].rolling(wy_win, min_periods=wy_min_periods).max().reset_index(level=0, drop=True)
    )
    df[f"tr_low_{wy_win}"] = (
        g["low"].rolling(wy_win, min_periods=wy_min_periods).min().reset_index(level=0, drop=True)
    )
    df[f"tr_high_{wy_win}_prev"] = (
        g["high"].shift(1).rolling(wy_win, min_periods=wy_min_periods).max().reset_index(level=0, drop=True)
    )
    df[f"tr_low_{wy_win}_prev"] = (
        g["low"].shift(1).rolling(wy_win, min_periods=wy_min_periods).min().reset_index(level=0, drop=True)
    )
    tr_span = df[f"tr_high_{wy_win}"] - df[f"tr_low_{wy_win}"]
    tr_span_safe = tr_span.replace(0, np.nan)
    tr_low_safe = df[f"tr_low_{wy_win}"].replace(0, np.nan)
    df[f"tr_width_{wy_win}"] = tr_span / tr_low_safe
    df[f"tr_pos_{wy_win}"] = (df["close"] - df[f"tr_low_{wy_win}"]) / tr_span_safe

    if "amount" in df.columns:
        df["amount_ma20"] = g["amount"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
        df["amount_ratio20"] = df["amount"] / (df["amount_ma20"] + 1e-9)
        wy_vol_ratio = df["amount_ratio20"]
    elif "volume" in df.columns:
        df["volume_ma20"] = g["volume"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
        df["volume_ratio20"] = df["volume"] / (df["volume_ma20"] + 1e-9)
        wy_vol_ratio = df["volume_ratio20"]
    else:
        wy_vol_ratio = pd.Series(np.nan, index=df.index)

    tr_high_prev = df[f"tr_high_{wy_win}_prev"]
    tr_low_prev = df[f"tr_low_{wy_win}_prev"]
    downtrend_ctx = (
        pd.notna(df["close"]) & pd.notna(df["ma20"]) & pd.notna(df["ma60"])
        & (df["close"] < df["ma20"]) & (df["ma20"] <= df["ma60"])
    )
    bottom_zone = df["close"] <= tr_low_prev * 1.06
    heavy_range = df["bar_range_pct"] >= (df["bar_range_pct_ma20"] * 1.25)
    ultra_range = df["bar_range_pct"] >= (df["bar_range_pct_ma20"] * 1.60)
    close_recovery_mid = df["close_pos"] >= 0.45
    close_recovery_strong = df["close_pos"] >= 0.60
    df[f"ps_{wy_win}"] = (
        downtrend_ctx
        & bottom_zone
        & heavy_range
        & (wy_vol_ratio >= 1.25)
        & close_recovery_mid
    )
    df[f"sc_{wy_win}"] = (
        downtrend_ctx
        & ((df["low"] < tr_low_prev * (1 - 0.004)) | df["ll"].fillna(False))
        & ultra_range
        & (wy_vol_ratio >= 1.60)
        & close_recovery_strong
    )
    sc_recent_8 = (
        g[f"sc_{wy_win}"]
        .transform(lambda s: s.shift(1).rolling(8, min_periods=1).max())
        .fillna(0)
        > 0
    )
    short_high_5 = g["high"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).max())
    df[f"ar_{wy_win}"] = (
        sc_recent_8
        & (df["close"] >= df["open"])
        & (df["close"] > df["prev_close"])
        & (df["high"] >= short_high_5)
        & (wy_vol_ratio >= 0.90)
    )
    df["sc_low_marker"] = np.where(df[f"sc_{wy_win}"], df["low"], np.nan)
    df["last_sc_low"] = g["sc_low_marker"].ffill()
    sc_recent_20 = (
        g[f"sc_{wy_win}"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=1).max())
        .fillna(0)
        > 0
    )
    test_zone = (
        (df["low"] >= df["last_sc_low"] * 0.985)
        & (df["low"] <= df["last_sc_low"] * 1.03)
    )
    df[f"st_{wy_win}"] = (
        sc_recent_20
        & test_zone
        & (wy_vol_ratio <= 1.05)
        & close_recovery_mid
        & (~df[f"sc_{wy_win}"])
    )
    df[f"spring_{wy_win}"] = (
        (df["low"] < tr_low_prev * (1 - 0.002))
        & (df["close"] > tr_low_prev)
        & (wy_vol_ratio >= 1.2)
    )
    df[f"tso_{wy_win}"] = (
        (df[f"sc_{wy_win}"] | df[f"spring_{wy_win}"])
        & (df["low"] < tr_low_prev * (1 - 0.010))
        & (wy_vol_ratio >= 1.80)
        & (df["close_pos"] >= 0.62)
    )
    df[f"utad_{wy_win}"] = (
        (df["high"] > tr_high_prev * (1 + 0.002))
        & (df["close"] < tr_high_prev)
        & (wy_vol_ratio >= 1.2)
    )
    df[f"sos_{wy_win}"] = (
        (df["close"] > tr_high_prev * (1 + 0.005))
        & (df["close"] >= df["open"])
        & (wy_vol_ratio >= 1.2)
    )
    recent_high_10 = g["high"].transform(lambda s: s.shift(1).rolling(10, min_periods=4).max())
    df[f"joc_{wy_win}"] = (
        (df[f"sos_{wy_win}"] | df[f"ar_{wy_win}"])
        & (df["close"] > tr_high_prev * (1 + 0.010))
        & (df["high"] >= recent_high_10)
        & (wy_vol_ratio >= 1.35)
        & (df["bar_range_pct"] >= df["bar_range_pct_ma20"] * 1.15)
        & (df["close_pos"] >= 0.62)
    )
    df[f"sow_{wy_win}"] = (
        (df["close"] < tr_low_prev * (1 - 0.005))
        & (df["close"] <= df["open"])
        & (wy_vol_ratio >= 1.2)
    )
    df[f"lps_{wy_win}"] = (
        (df["close"] >= tr_high_prev * (1 - 0.01))
        & (df["close"] <= tr_high_prev * (1 + 0.03))
        & (wy_vol_ratio <= 1.0)
        & df["hl"].fillna(False)
    )
    df[f"lpsy_{wy_win}"] = (
        (df["close"] <= tr_low_prev * (1 + 0.01))
        & (df["close"] >= tr_low_prev * (1 - 0.03))
        & (wy_vol_ratio <= 1.0)
        & df["lh"].fillna(False)
    )

    return df


def _row_events(row: pd.Series, win: int = 60) -> str:
    events = []
    for key, name in [
        (f"ps_{win}", "PS"),
        (f"sc_{win}", "SC"),
        (f"ar_{win}", "AR"),
        (f"st_{win}", "ST"),
        (f"tso_{win}", "TSO"),
        (f"spring_{win}", "Spring"),
        (f"sos_{win}", "SOS"),
        (f"joc_{win}", "JOC"),
        (f"lps_{win}", "LPS"),
        (f"utad_{win}", "UTAD"),
        (f"sow_{win}", "SOW"),
        (f"lpsy_{win}", "LPSY"),
    ]:
        if bool(row.get(key, False)):
            events.append(name)
    return " / ".join(events) if events else "无"


def _classify_wyckoff_phase_row(row: pd.Series, win: int = 60) -> str:
    up_score = int(bool(row.get("hh"))) + int(bool(row.get("hl"))) + int(bool(row.get("hc")))
    down_score = int(bool(row.get("lh"))) + int(bool(row.get("ll"))) + int(bool(row.get("lc")))
    tr_width = row.get(f"tr_width_{win}")
    tr_pos = row.get(f"tr_pos_{win}")

    in_range = pd.notna(tr_width) and 0.08 <= tr_width <= 0.55
    ma20 = row.get("ma20")
    ma60 = row.get("ma60")
    close = row.get("close")
    up_trend = pd.notna(close) and pd.notna(ma20) and pd.notna(ma60) and close > ma20 > ma60
    down_trend = pd.notna(close) and pd.notna(ma20) and pd.notna(ma60) and close < ma20 < ma60

    if bool(row.get(f"sc_{win}", False)) or bool(row.get(f"ps_{win}", False)):
        return "A阶段-止跌初期"
    if bool(row.get(f"ar_{win}", False)):
        return "A阶段-止跌初期"
    if bool(row.get(f"st_{win}", False)) and in_range:
        return "B阶段-吸筹震荡"
    if bool(row.get(f"tso_{win}", False)):
        return "C阶段-Spring测试"
    if bool(row.get(f"joc_{win}", False)) and in_range:
        return "D阶段-上破准备"
    if bool(row.get(f"sos_{win}", False)) and up_trend and up_score >= 2:
        return "E阶段-拉升(Markup)"
    if bool(row.get(f"sow_{win}", False)) and down_trend and down_score >= 2:
        return "E阶段-下跌(Markdown)"
    if bool(row.get(f"spring_{win}", False)):
        return "C阶段-Spring测试"
    if bool(row.get(f"utad_{win}", False)):
        return "C阶段-UTAD"
    if in_range:
        if pd.notna(tr_pos) and tr_pos <= 0.45:
            return "B阶段-吸筹震荡"
        if pd.notna(tr_pos) and tr_pos >= 0.55:
            return "B阶段-派发震荡"
        return "阶段未明"
    if up_score >= 2 and pd.notna(close) and pd.notna(ma20) and close >= ma20:
        return "D阶段-上破准备"
    if down_score >= 2 and pd.notna(close) and pd.notna(ma20) and close <= ma20:
        return "D阶段-下破准备"
    if up_score == 0 and down_score >= 2:
        return "A阶段-见顶初期"
    if down_score == 0 and up_score >= 2:
        return "A阶段-止跌初期"
    return "阶段未明"


def enrich_wyckoff_latest(latest: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    if latest is None or latest.empty:
        return latest
    out = latest.copy()
    for col in ("hh", "hl", "hc", "lh", "ll", "lc"):
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False)
    out["structure_up_score"] = out[["hh", "hl", "hc"]].sum(axis=1).astype(int)
    out["structure_hhh"] = out[["hh", "hl", "hc"]].apply(
        lambda r: "/".join([k.upper() for k in ("hh", "hl", "hc") if bool(r[k])]) or "-",
        axis=1,
    )
    out["wyckoff_signal"] = out.apply(lambda r: _row_events(r, win=win), axis=1)
    out["wyckoff_phase"] = out.apply(lambda r: _classify_wyckoff_phase_row(r, win=win), axis=1)
    return out


def compute_wyckoff_sequence_features(df: pd.DataFrame, win: int = 60, lookback: int = 120) -> pd.DataFrame:
    if df is None or df.empty or "code" not in df.columns:
        return pd.DataFrame(columns=["code", "wy_event_count", "wy_events_present", "wy_sequence_ok"])
    lookback = max(30, int(lookback))
    event_prefixes = ["ps", "sc", "ar", "st", "tso", "spring", "sos", "joc", "lps", "utad", "sow", "lpsy"]
    label_map = {
        "ps": "PS",
        "sc": "SC",
        "ar": "AR",
        "st": "ST",
        "tso": "TSO",
        "spring": "Spring",
        "sos": "SOS",
        "joc": "JOC",
        "lps": "LPS",
        "utad": "UTAD",
        "sow": "SOW",
        "lpsy": "LPSY",
    }
    rows = []
    ordered = df.sort_values(["code", "date"])
    for code, g in ordered.groupby("code", sort=False):
        tail = g.tail(lookback).reset_index(drop=True)
        row = {"code": code}
        last_pos = {}
        for p in event_prefixes:
            col = f"{p}_{win}"
            if col in tail.columns:
                s = tail[col].fillna(False).astype(bool)
                has_evt = bool(s.any())
                row[f"has_{p}"] = has_evt
                if has_evt:
                    pos = np.flatnonzero(s.values)
                    last_pos[p] = int(pos[-1]) if len(pos) else -1
                else:
                    last_pos[p] = -1
            else:
                row[f"has_{p}"] = False
                last_pos[p] = -1

        core_seq = ["ps", "sc", "ar", "st", "spring", "sos", "joc", "lps"]
        has_core = all(last_pos[p] >= 0 for p in core_seq)
        in_order = has_core and all(last_pos[core_seq[i]] < last_pos[core_seq[i + 1]] for i in range(len(core_seq) - 1))
        row["wy_sequence_ok"] = bool(in_order)
        row["wy_event_count"] = int(sum(bool(row[f"has_{p}"]) for p in event_prefixes))
        row["wy_events_present"] = " / ".join([label_map[p] for p in event_prefixes if bool(row[f"has_{p}"])]) or "无"
        rows.append(row)
    return pd.DataFrame(rows)


def merge_meta(price_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    if meta_df is None or meta_df.empty:
        return price_df
    if "code" not in meta_df.columns:
        return price_df
    meta_df = meta_df.copy()
    meta_df["code"] = normalize_code(meta_df["code"])
    meta_df = meta_df.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="last")
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
    api_df = normalize_columns(api_df).copy()
    if "code" not in api_df.columns:
        return meta_df
    api_df["code"] = normalize_code(api_df["code"])
    api_df = api_df.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="last")
    if meta_df is None or meta_df.empty:
        return api_df
    meta_df = meta_df.copy()
    if "code" not in meta_df.columns:
        return api_df
    meta_df["code"] = normalize_code(meta_df["code"])
    meta_df = meta_df.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="last")
    # Use outer merge so API can add metadata for codes not present in uploaded meta.
    meta_df = meta_df.merge(api_df, on="code", how="outer", suffixes=("", "_api"))
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
    elif "industry" in s.columns and "return_5d" in s.columns:
        sector_ret = s.groupby("industry")["return_5d"].transform("mean")
        s["sector_score"] = sector_ret.rank(pct=True)
    elif "board" in s.columns and "return_5d" in s.columns:
        sector_ret = s.groupby("board")["return_5d"].transform("mean")
        s["sector_score"] = sector_ret.rank(pct=True)
    else:
        s["sector_score"] = 0.5
        warnings.append("缺少题材强度所需字段(sector_return_5d 或 return_5d+board/industry)，使用中性得分。")

    win = int(params.get("trend_window", 20))
    dd_field = f"drawdown_{win}"
    if params.get("drawdown_mode") in ("max_drawdown", "最大回撤") and f"max_drawdown_{win}" in s.columns:
        dd_field = f"max_drawdown_{win}"
    if dd_field not in s.columns:
        warnings.append("缺少回撤字段，回撤评分记为0。")
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

    if params.get("enable_wyckoff_event_filter"):
        evt_map = {
            "PS": "ps",
            "SC": "sc",
            "AR": "ar",
            "ST": "st",
            "TSO": "tso",
            "Spring": "spring",
            "SOS": "sos",
            "JOC": "joc",
            "LPS": "lps",
            "UTAD": "utad",
            "SOW": "sow",
            "LPSY": "lpsy",
        }
        req_events = params.get("wy_required_events", []) or []
        for evt in req_events:
            prefix = evt_map.get(evt)
            if not prefix:
                continue
            col = f"has_{prefix}"
            if col in l1.columns:
                l1 = l1[l1[col].fillna(False)]
            else:
                warnings.append(f"缺少事件字段 {evt}，已跳过该项筛选。")
        if params.get("wy_require_sequence"):
            if "wy_sequence_ok" in l1.columns:
                l1 = l1[l1["wy_sequence_ok"].fillna(False)]
            else:
                warnings.append("缺少序列字段，已跳过序列完整筛选。")
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
    if params.get("drawdown_mode") in ("max_drawdown", "最大回撤") and f"max_drawdown_{win}" in l2.columns:
        dd_field = f"max_drawdown_{win}"
    if dd_field in l2.columns:
        cond_d = (l2[dd_field] >= params["dd_min"]) & (
            l2[dd_field] <= params["dd_max"]
        )
    else:
        cond_d = pd.Series(False, index=l2.index)
        warnings.append("缺少回撤字段，已跳过回撤健康条件。")
    l2["cond_a"] = cond_a.fillna(False)
    l2["cond_b"] = cond_b.fillna(False)
    l2["cond_c"] = cond_c.fillna(False)
    l2["cond_d"] = cond_d.fillna(False)

    cond_e = pd.Series(True, index=l2.index)
    if params.get("enable_turn_up"):
        ma20_slope_col = "ma20_slope_10" if "ma20_slope_10" in l2.columns else "ma20_slope_5"
        cond_e = (
            (l2["ma5_slope_3"] >= params.get("turn_up_ma5", 0))
            & (l2["ma10_slope_5"] >= params.get("turn_up_ma10", 0))
            & (l2[ma20_slope_col] >= params.get("turn_up_ma20", 0))
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
    if "structure_up_score" in out.columns:
        out["structure_up_score"] = out["structure_up_score"].fillna(0).astype(int)
    if "structure_hhh" in out.columns:
        out["structure_hhh"] = out["structure_hhh"].fillna("-")
    if "wyckoff_phase" in out.columns:
        out["wyckoff_phase"] = out["wyckoff_phase"].fillna("阶段未明")
    if "wyckoff_signal" in out.columns:
        out["wyckoff_signal"] = out["wyckoff_signal"].fillna("无")
    if "wy_events_present" in out.columns:
        out["wy_events_present"] = out["wy_events_present"].fillna("无")
    if "wy_sequence_ok" in out.columns:
        out["wy_sequence_ok"] = out["wy_sequence_ok"].map(lambda x: "是" if bool(x) else "否")

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
        "structure_hhh",
        "structure_up_score",
        "wyckoff_phase",
        "wyckoff_signal",
        "wy_events_present",
        "wy_sequence_ok",
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
    names = make_display_names(win)
    cols_color = [names[f"return_{win}d_pct"], names["buy_distance_pct"]]
    for col in cols_color:
        if col in df.columns:
            styler = styler.applymap(_color, subset=[col])

    formatters = {}
    if names["close"] in df.columns:
        formatters[names["close"]] = "{:.2f}"
    if names[f"high_{win}d_price"] in df.columns:
        formatters[names[f"high_{win}d_price"]] = "{:.2f}"
    if names["ideal_buy_price"] in df.columns:
        formatters[names["ideal_buy_price"]] = "{:.2f}"
    if names[f"return_{win}d_pct"] in df.columns:
        formatters[names[f"return_{win}d_pct"]] = "{:.2f}%"
    if names[f"drawdown_{win}_pct"] in df.columns:
        formatters[names[f"drawdown_{win}_pct"]] = "{:.2f}%"
    if names[f"max_drawdown_{win}_pct"] in df.columns:
        formatters[names[f"max_drawdown_{win}_pct"]] = "{:.2f}%"
    if names["buy_distance_pct"] in df.columns:
        formatters[names["buy_distance_pct"]] = "{:.2f}%"
    if names["score_pct"] in df.columns:
        formatters[names["score_pct"]] = "{:.2f}"
    if names["amount_5d_yi"] in df.columns:
        formatters[names["amount_5d_yi"]] = "{:.2f}"
    if names[f"volatility_{win}_pct"] in df.columns:
        formatters[names[f"volatility_{win}_pct"]] = "{:.2f}%"
    if names[f"up_down_ratio_{win}"] in df.columns:
        formatters[names[f"up_down_ratio_{win}"]] = "{:.2f}"
    if names[f"drawdown_days_{win}"] in df.columns:
        formatters[names[f"drawdown_days_{win}"]] = "{:.0f}"
    if names["structure_up_score"] in df.columns:
        formatters[names["structure_up_score"]] = "{:.0f}"

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
    if df is None or df.empty:
        st.info("暂无数据")
        return
    if use_aggrid:
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(sortable=True, filter=True, resizable=True, editable=False)
            gb.configure_grid_options(domLayout="normal")
            grid_options = gb.build()
            AgGrid(
                df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.NO_UPDATE,
                fit_columns_on_grid_load=False,
                height=height,
                theme="streamlit",
                reload_data=False,
            )
            return
        except Exception:
            st.warning("交互表格不可用，已回退到基础表格。")
    st.dataframe(
        style_dataframe(df, up_red=up_red, trend_window=trend_window),
        use_container_width=True,
        hide_index=True,
        height=height,
    )

def plot_kline(df: pd.DataFrame, code: str, up_red: bool = True) -> go.Figure:
    d = df[df["code"] == code].copy().sort_values("date")
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
            name="Kline",
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

    event_styles = [
        ("ps_60", "PS", "#8d6e63", "diamond", "low", -0.004),
        ("sc_60", "SC", "#d32f2f", "x", "low", -0.006),
        ("ar_60", "AR", "#43a047", "triangle-up", "high", 0.006),
        ("st_60", "ST", "#1e88e5", "circle", "low", -0.004),
        ("tso_60", "TSO", "#ad1457", "x-thin", "low", -0.008),
        ("spring_60", "Spring", "#6d4c41", "triangle-down", "low", -0.006),
        ("sos_60", "SOS", "#2e7d32", "triangle-up", "high", 0.006),
        ("joc_60", "JOC", "#00897b", "diamond", "high", 0.008),
        ("lps_60", "LPS", "#00acc1", "circle-open", "low", -0.004),
        ("utad_60", "UTAD", "#ef6c00", "triangle-down", "high", 0.006),
        ("sow_60", "SOW", "#6a1b9a", "triangle-down", "low", -0.006),
        ("lpsy_60", "LPSY", "#3949ab", "circle-open", "high", 0.004),
    ]
    for col, label, color, symbol, anchor_col, offset in event_styles:
        if col not in d.columns:
            continue
        m = d[col].fillna(False).astype(bool)
        if not m.any():
            continue
        base = d.loc[m, anchor_col]
        y = base * (1 + offset)
        fig.add_trace(
            go.Scatter(
                x=d.loc[m, "date"],
                y=y,
                mode="markers",
                name=label,
                marker=dict(color=color, size=9, symbol=symbol, line=dict(width=1, color=color)),
                hovertemplate=f"{label}<extra></extra>",
            )
        )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
    )
    return fig

def build_suggestions(layers: dict, params: dict, warnings: List[str], price_df: pd.DataFrame, meta_df: pd.DataFrame) -> List[str]:
    tips = []
    win = int(params.get("trend_window", 20))
    if params.get("enable_wyckoff_event_filter") and len(layers["layer1"]) == 0:
        tips.append("威科夫事件筛选后为空：可减少必须事件或关闭8步序列完整要求。")
    if "name" not in price_df.columns and (meta_df is None or meta_df.empty or "name" not in meta_df.columns):
        tips.append("未发现股票名称字段：可上传元数据CSV或通过API补充名称。")
    if len(layers["layer2"]) == 0:
        tips.append(f"第2层结果为空：可适当降低{win}日涨幅下限或提高Top N。")
    if len(layers["layer3"]) == 0:
        tips.append("第3层结果为空：可降低趋势条件最少满足数或放宽回撤区间。")
    if len(layers["layer4"]) == 0:
        tips.append("第4层结果为空：可适当放宽买点距离上下限。")
    if len(layers["layer5"]) == 0:
        tips.append("最终Top为空：请检查数据完整性或放宽筛选条件。")
    if any("成交量" in w or "成交额" in w for w in warnings):
        tips.append("缺少成交量/成交额会影响量价判断，建议补齐。")
    if params.get("enable_board") and "board" not in price_df.columns:
        tips.append("当前无板块字段，板块过滤已跳过。可上传元数据补齐板块。")
    return tips


def build_wyckoff_phase_pool(
    latest: pd.DataFrame,
    layers: dict,
    phase_scope: str = "All symbols",
    phases: Optional[List[str]] = None,
    required_events: Optional[List[str]] = None,
) -> pd.DataFrame:
    if latest is None or latest.empty:
        return pd.DataFrame()

    base = latest.copy()
    scope = str(phase_scope or "All symbols")
    if scope == "Layer1" and "layer1" in layers and "code" in layers["layer1"].columns:
        codes = set(layers["layer1"]["code"].dropna().astype(str))
        base = base[base["code"].astype(str).isin(codes)]
    elif scope == "Layer4 candidates" and "layer4" in layers and "code" in layers["layer4"].columns:
        codes = set(layers["layer4"]["code"].dropna().astype(str))
        base = base[base["code"].astype(str).isin(codes)]
    elif scope == "Final Top" and "layer5" in layers and "code" in layers["layer5"].columns:
        codes = set(layers["layer5"]["code"].dropna().astype(str))
        base = base[base["code"].astype(str).isin(codes)]

    if "wyckoff_phase" not in base.columns:
        base["wyckoff_phase"] = "阶段未明"
    base["wyckoff_phase"] = base["wyckoff_phase"].fillna("阶段未明")

    selected_phases = [x for x in (phases or []) if isinstance(x, str) and x.strip()]
    if selected_phases:
        base = base[base["wyckoff_phase"].isin(selected_phases)]

    for evt in (required_events or []):
        prefix = WYCKOFF_EVENT_PREFIX.get(str(evt))
        if not prefix:
            continue
        has_col = f"has_{prefix}"
        if has_col in base.columns:
            base = base[base[has_col].fillna(False)]

    if base.empty:
        return base
    return base.sort_values(["wyckoff_phase", "code"]).reset_index(drop=True)


def format_backtest_trades_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    out = trades_df.copy()
    for c in ["signal_date", "entry_date", "exit_date"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date
    if "exit_reason" in out.columns:
        out["exit_reason"] = out["exit_reason"].map(BACKTEST_EXIT_REASON_LABELS).fillna(out["exit_reason"])

    ordered = [c for c in BACKTEST_TRADE_COL_LABELS.keys() if c in out.columns]
    tail = [c for c in out.columns if c not in ordered]
    out = out[ordered + tail]
    return out.rename(columns={k: v for k, v in BACKTEST_TRADE_COL_LABELS.items() if k in out.columns})


def to_json_compatible(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (dt.date, dt.datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (list, tuple, set)):
        return [to_json_compatible(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_compatible(v) for k, v in obj.items()}
    return str(obj)


def collect_sidebar_settings_snapshot(params: dict) -> Dict:
    banned = {
        "loaded_price_df",
        "analysis_cache",
        "api_meta_df",
        "price_csv_file",
        "meta_csv_file",
    }
    snapshot = {}
    for key in sorted(st.session_state.keys()):
        if key.startswith("_") or key in banned:
            continue
        value = st.session_state.get(key)
        if isinstance(value, (pd.DataFrame, pd.Series, io.BytesIO, bytes, bytearray)):
            continue
        if callable(value):
            continue
        snapshot[key] = to_json_compatible(value)
    snapshot["analysis_params"] = to_json_compatible(params or {})
    return snapshot


def build_backtest_report_zip(
    trades_view: pd.DataFrame,
    eq_df: pd.DataFrame,
    bt_metrics: Dict[str, float],
    bt_notes: List[str],
    params: dict,
    sidebar_snapshot: Dict,
) -> bytes:
    buf = io.BytesIO()
    created_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        tv = trades_view.copy() if trades_view is not None else pd.DataFrame()
        if not tv.empty:
            zf.writestr("trades.csv", tv.to_csv(index=False, encoding="utf-8-sig"))
        else:
            zf.writestr("trades.csv", "")

        eq_export = eq_df.copy() if eq_df is not None else pd.DataFrame()
        if not eq_export.empty and "date" in eq_export.columns:
            eq_export["date"] = pd.to_datetime(eq_export["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        zf.writestr("equity_curve.csv", eq_export.to_csv(index=False, encoding="utf-8-sig"))

        zf.writestr(
            "metrics.json",
            json.dumps(to_json_compatible(bt_metrics or {}), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            "backtest_params.json",
            json.dumps(to_json_compatible(params or {}), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            "sidebar_settings_snapshot.json",
            json.dumps(to_json_compatible(sidebar_snapshot or {}), ensure_ascii=False, indent=2),
        )
        zf.writestr("notes.txt", "\n".join(bt_notes or []))
        summary = [
            "# Wyckoff Backtest Report",
            f"- created_at: {created_at}",
            f"- total_trades: {int((bt_metrics or {}).get('total_trades', 0))}",
            f"- win_rate_pct: {float((bt_metrics or {}).get('win_rate_pct', 0.0)):.2f}",
            f"- cum_return_pct: {float((bt_metrics or {}).get('cum_return_pct', 0.0)):.2f}",
            f"- max_drawdown_pct: {float((bt_metrics or {}).get('max_drawdown_pct', 0.0)):.2f}",
            "",
            "Files:",
            "- trades.csv",
            "- equity_curve.csv",
            "- metrics.json",
            "- backtest_params.json",
            "- sidebar_settings_snapshot.json",
            "- notes.txt",
        ]
        zf.writestr("README.md", "\n".join(summary))
    return buf.getvalue()


def run_wyckoff_backtest(
    df: pd.DataFrame,
    entry_events: List[str],
    exit_events: List[str],
    stop_loss: float,
    take_profit: float,
    max_hold_bars: int,
    lookback_bars: int,
    range_mode: str = "lookback_bars",
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    fee_bps: float = 8.0,
    cooldown_bars: int = 0,
    code_pool: Optional[List[str]] = None,
    win: int = 60,
    initial_capital: float = 1_000_000.0,
    position_pct: float = 0.20,
    risk_per_trade: float = 0.01,
    max_positions: int = 5,
    position_mode: str = "min",
    prioritize_signals: bool = True,
    enforce_t1: bool = True,
    priority_mode: str = "balanced",
    priority_topk_per_day: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:
    notes: List[str] = []
    initial_capital = max(1.0, float(initial_capital))
    empty_metrics = {
        "initial_capital": float(initial_capital),
        "final_equity": float(initial_capital),
        "total_trades": 0.0,
        "skipped_trades": 0.0,
        "fill_rate_pct": 0.0,
        "max_concurrent_positions": 0.0,
        "win_rate_pct": 0.0,
        "avg_ret_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "profit_factor": 0.0,
        "payoff_ratio": 0.0,
        "cum_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
    }
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), empty_metrics, notes

    req_cols = {"code", "date", "open", "high", "low", "close"}
    miss = [c for c in req_cols if c not in df.columns]
    if miss:
        notes.append(f"回测缺少字段：{','.join(miss)}。")
        return pd.DataFrame(), pd.DataFrame(), empty_metrics, notes

    event_map = {
        "PS": "ps",
        "SC": "sc",
        "AR": "ar",
        "ST": "st",
        "TSO": "tso",
        "Spring": "spring",
        "SOS": "sos",
        "JOC": "joc",
        "LPS": "lps",
        "UTAD": "utad",
        "SOW": "sow",
        "LPSY": "lpsy",
    }

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values(["code", "date"])
    if code_pool:
        pool = set(code_pool)
        work = work[work["code"].isin(pool)]

    range_mode = str(range_mode or "lookback_bars").strip().lower()
    if range_mode not in {"lookback_bars", "custom_dates"}:
        range_mode = "lookback_bars"

    if range_mode == "custom_dates":
        start_ts = pd.to_datetime(start_date, errors="coerce") if start_date is not None else pd.NaT
        end_ts = pd.to_datetime(end_date, errors="coerce") if end_date is not None else pd.NaT
        if pd.notna(start_ts) and pd.notna(end_ts) and start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
        if pd.notna(start_ts):
            work = work[work["date"] >= start_ts]
        if pd.notna(end_ts):
            work = work[work["date"] <= end_ts]
    elif lookback_bars and lookback_bars > 0:
        work = work.groupby("code", group_keys=False).tail(int(lookback_bars) + 30)
    if work.empty:
        notes.append("回测范围内没有可用数据。")
        return pd.DataFrame(), pd.DataFrame(), empty_metrics, notes

    def _event_cols(evts: List[str]) -> Tuple[List[str], List[str]]:
        cols, missing = [], []
        for evt in evts:
            prefix = event_map.get(evt)
            if not prefix:
                missing.append(evt)
                continue
            col = f"{prefix}_{win}"
            if col in work.columns:
                cols.append(col)
            else:
                missing.append(evt)
        return cols, missing

    entry_cols, missing_entry = _event_cols(entry_events)
    exit_cols, missing_exit = _event_cols(exit_events)
    if missing_entry:
        notes.append(f"入场事件缺失：{','.join(missing_entry)}。")
    if missing_exit:
        notes.append(f"离场事件缺失：{','.join(missing_exit)}。")
    if not entry_cols:
        notes.append("没有可用入场事件列，回测未执行。")
        return pd.DataFrame(), pd.DataFrame(), empty_metrics, notes

    stop_loss = max(0.0, float(stop_loss))
    take_profit = max(0.0, float(take_profit))
    max_hold_bars = max(2, int(max_hold_bars))
    cooldown_bars = max(0, int(cooldown_bars))
    fee_cost = max(0.0, float(fee_bps)) / 10000.0
    position_pct = min(1.0, max(0.0, float(position_pct)))
    risk_per_trade = min(1.0, max(0.0, float(risk_per_trade)))
    max_positions = max(1, int(max_positions))
    prioritize_signals = bool(prioritize_signals)
    enforce_t1 = bool(enforce_t1)
    priority_mode = str(priority_mode or "balanced").strip().lower()
    if priority_mode not in {"phase_first", "balanced", "momentum"}:
        priority_mode = "balanced"
    priority_topk_per_day = max(0, int(priority_topk_per_day))
    priority_mode_label = BACKTEST_PRIORITY_MODE_LABELS.get(priority_mode, priority_mode)
    position_mode = str(position_mode or "min").strip().lower()
    if position_mode not in {"fixed", "risk", "min"}:
        notes.append(f"未知仓位模式({position_mode})，已回退为 min。")
        position_mode = "min"

    ret_cols = [c for c in work.columns if re.fullmatch(r"return_\d+d", str(c))]
    ret_ref_col = None
    if ret_cols:
        def _ret_dist(col: str) -> int:
            m = re.search(r"return_(\d+)d", col)
            n = int(m.group(1)) if m else 999
            return abs(n - 20)
        ret_ref_col = sorted(ret_cols, key=_ret_dist)[0]

    vol_cols = [c for c in work.columns if re.fullmatch(r"volatility_\d+", str(c))]
    vol_ref_col = None
    if vol_cols:
        def _vol_dist(col: str) -> int:
            m = re.search(r"volatility_(\d+)", col)
            n = int(m.group(1)) if m else 999
            return abs(n - 20)
        vol_ref_col = sorted(vol_cols, key=_vol_dist)[0]

    def _phase_priority_score(phase: str) -> float:
        p = str(phase or "")
        mapping = {
            "C阶段-Spring测试": 2.2,
            "D阶段-上破准备": 2.0,
            "B阶段-吸筹震荡": 1.4,
            "A阶段-止跌初期": 1.1,
            "E阶段-拉升(Markup)": 0.2,
            "阶段未明": 0.0,
            "A阶段-见顶初期": -0.8,
            "B阶段-派发震荡": -1.2,
            "C阶段-UTAD": -2.0,
            "D阶段-下破准备": -2.2,
            "E阶段-下跌(Markdown)": -2.5,
        }
        return float(mapping.get(p, 0.0))

    def _build_mark_to_market_curve(
        trades: pd.DataFrame,
        fallback_start: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if trades is None or trades.empty:
            if fallback_start is None or pd.isna(fallback_start):
                return pd.DataFrame()
            base_dt = pd.to_datetime(fallback_start)
            return pd.DataFrame(
                [
                    {
                        "date": base_dt,
                        "cash": float(initial_capital),
                        "position_value": 0.0,
                        "equity": float(initial_capital),
                        "realized_equity": float(initial_capital),
                        "positions": 0,
                    }
                ]
            )

        px = work[["code", "date", "close"]].copy()
        px["close"] = pd.to_numeric(px["close"], errors="coerce")
        px = px.dropna(subset=["date"]).sort_values(["code", "date"])
        if px.empty:
            return pd.DataFrame()

        start_dt = pd.to_datetime(trades["entry_date"].min())
        end_dt = pd.to_datetime(trades["exit_date"].max())
        timeline = pd.to_datetime(
            px.loc[(px["date"] >= start_dt) & (px["date"] <= end_dt), "date"]
            .drop_duplicates()
            .sort_values()
        )
        if timeline.empty:
            timeline = pd.to_datetime(
                pd.concat([trades["entry_date"], trades["exit_date"]], ignore_index=True)
                .dropna()
                .drop_duplicates()
                .sort_values()
            )
        if timeline.empty:
            return pd.DataFrame()

        timeline_index = pd.DatetimeIndex(timeline)
        traded_codes = set(trades["code"].dropna().astype(str).tolist())
        close_map: Dict[str, pd.Series] = {}
        px_traded = px[px["code"].astype(str).isin(traded_codes)]
        for code, g in px_traded.groupby("code", sort=False):
            s = g.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if not s.empty:
                close_map[str(code)] = s.reindex(timeline_index, method="ffill")

        entry_map: Dict[pd.Timestamp, List[Dict]] = {}
        exit_map: Dict[pd.Timestamp, List[Dict]] = {}
        for idx, t in trades.reset_index(drop=True).iterrows():
            entry_dt = pd.to_datetime(t["entry_date"])
            exit_dt = pd.to_datetime(t["exit_date"])
            pos = {
                "id": int(idx),
                "code": str(t["code"]),
                "shares": int(t["shares"]),
                "entry_price": float(t["entry_price"]),
                "position_value": float(t["position_value"]),
                "exit_value": float(t["exit_value"]),
                "pnl_amount": float(t["pnl_amount"]),
            }
            entry_map.setdefault(entry_dt, []).append(pos)
            exit_map.setdefault(exit_dt, []).append(pos)

        cash_curve = float(initial_capital)
        realized_equity_curve = float(initial_capital)
        active_curve: Dict[int, Dict] = {}
        rows = []

        for cur_dt in timeline_index:
            for pos in exit_map.get(cur_dt, []):
                pid = int(pos["id"])
                if pid in active_curve:
                    cash_curve += float(pos["exit_value"])
                    realized_equity_curve += float(pos["pnl_amount"])
                    del active_curve[pid]

            for pos in entry_map.get(cur_dt, []):
                pid = int(pos["id"])
                if pid not in active_curve:
                    cash_curve -= float(pos["position_value"])
                    active_curve[pid] = pos

            mtm_val = 0.0
            for pos in active_curve.values():
                px_series = close_map.get(str(pos["code"]))
                px_now = np.nan if px_series is None else px_series.loc[cur_dt]
                if not np.isfinite(px_now) or float(px_now) <= 0:
                    px_now = float(pos["entry_price"])
                mtm_val += float(pos["shares"]) * float(px_now)

            rows.append(
                {
                    "date": pd.to_datetime(cur_dt),
                    "cash": float(cash_curve),
                    "position_value": float(mtm_val),
                    "equity": float(cash_curve + mtm_val),
                    "realized_equity": float(realized_equity_curve),
                    "positions": int(len(active_curve)),
                }
            )
        return pd.DataFrame(rows)

    # Step 1: generate per-symbol candidate trades from signals.
    candidates = []
    t1_no_sellbar_skips = 0
    for code, g in work.groupby("code", sort=False):
        x = g.reset_index(drop=True)
        n = len(x)
        if n < 3:
            continue
        entry_sig = x[entry_cols].fillna(False).astype(bool).any(axis=1)
        exit_sig = x[exit_cols].fillna(False).astype(bool).any(axis=1) if exit_cols else pd.Series(False, index=x.index)
        i = 0
        while i < n - 1:
            if not bool(entry_sig.iat[i]):
                i += 1
                continue
            entry_idx = i + 1
            if entry_idx >= n:
                break
            entry_open = pd.to_numeric(x.at[entry_idx, "open"], errors="coerce")
            if pd.isna(entry_open) or entry_open <= 0:
                i += 1
                continue

            signal_hits = []
            for evt in entry_events:
                p = event_map.get(evt)
                if not p:
                    continue
                col = f"{p}_{win}"
                if col in x.columns and bool(x.at[i, col]):
                    signal_hits.append(evt)
            signal_tag = " / ".join(signal_hits) if signal_hits else "Signal"

            event_weight = float(sum(BACKTEST_ENTRY_EVENT_WEIGHTS.get(evt, 1.0) for evt in signal_hits))
            phase_label = str(x.at[i, "wyckoff_phase"]) if "wyckoff_phase" in x.columns else ""
            if not phase_label or phase_label.lower() == "nan":
                try:
                    phase_label = str(_classify_wyckoff_phase_row(x.loc[i], win=win))
                except Exception:
                    phase_label = "阶段未明"
            phase_score = _phase_priority_score(phase_label)
            structure_score = 0
            for c in ("hh", "hl", "hc"):
                if c in x.columns and bool(x.at[i, c]):
                    structure_score += 1

            volume_signal = 0.0
            vol_col = "amount_ratio20" if "amount_ratio20" in x.columns else ("volume_ratio20" if "volume_ratio20" in x.columns else "")
            if vol_col:
                v = pd.to_numeric(x.at[i, vol_col], errors="coerce")
                if pd.notna(v):
                    v_clip = float(np.clip(v, 0.2, 4.0))
                    if v_clip >= 1.0:
                        volume_signal = min(1.6, (v_clip - 1.0) * 0.9)
                    else:
                        volume_signal = -min(0.8, (1.0 - v_clip) * 1.2)

            trend_signal = 0.0
            trend_phase_adj = 0.0
            if ret_ref_col and ret_ref_col in x.columns:
                rv = pd.to_numeric(x.at[i, ret_ref_col], errors="coerce")
                if pd.notna(rv):
                    rv = float(rv)
                    trend_signal = float(np.clip(rv, -0.25, 0.70) * 6.0)
                    if rv <= -0.12:
                        trend_phase_adj = -1.2
                    elif rv < 0.0:
                        trend_phase_adj = -0.3
                    elif rv <= 0.20:
                        trend_phase_adj = 1.2
                    elif rv <= 0.45:
                        trend_phase_adj = 0.5
                    else:
                        trend_phase_adj = -0.6

            volatility_penalty = 0.0
            if vol_ref_col and vol_ref_col in x.columns:
                vv = pd.to_numeric(x.at[i, vol_ref_col], errors="coerce")
                if pd.notna(vv):
                    vv = float(vv)
                    if vv >= 0.12:
                        volatility_penalty = -1.2
                    elif vv >= 0.08:
                        volatility_penalty = -0.6
                    elif vv <= 0.01:
                        volatility_penalty = -0.2

            if priority_mode == "phase_first":
                trend_component = float(trend_phase_adj)
                quality_score = float(
                    phase_score * 2.6
                    + event_weight * 1.5
                    + structure_score * 0.7
                    + trend_phase_adj * 1.2
                    + volume_signal * 0.5
                    + volatility_penalty * 0.9
                )
            elif priority_mode == "momentum":
                trend_component = float(trend_signal)
                quality_score = float(
                    phase_score * 0.8
                    + event_weight * 1.6
                    + structure_score * 0.8
                    + trend_signal * 1.4
                    + volume_signal * 0.8
                    + volatility_penalty * 0.4
                )
            else:
                trend_component = float(0.5 * trend_signal + 0.5 * trend_phase_adj)
                quality_score = float(
                    phase_score * 1.8
                    + event_weight * 1.8
                    + structure_score * 0.8
                    + trend_component
                    + volume_signal * 0.6
                    + volatility_penalty * 0.8
                )

            stop_px = entry_open * (1 - stop_loss) if stop_loss > 0 else np.nan
            take_px = entry_open * (1 + take_profit) if take_profit > 0 else np.nan
            exit_idx = None
            exit_px = None
            exit_reason = None
            entry_trade_date = pd.to_datetime(x.at[entry_idx, "date"], errors="coerce")
            last_sellable_idx = None

            j = entry_idx
            while j < n:
                bar_date = pd.to_datetime(x.at[j, "date"], errors="coerce")
                sellable_today = True
                if enforce_t1:
                    sellable_today = bool(pd.notna(bar_date) and pd.notna(entry_trade_date) and (bar_date > entry_trade_date))
                if not sellable_today:
                    j += 1
                    continue
                last_sellable_idx = j

                high_j = pd.to_numeric(x.at[j, "high"], errors="coerce")
                low_j = pd.to_numeric(x.at[j, "low"], errors="coerce")
                close_j = pd.to_numeric(x.at[j, "close"], errors="coerce")

                stop_hit = stop_loss > 0 and pd.notna(low_j) and low_j <= stop_px
                take_hit = take_profit > 0 and pd.notna(high_j) and high_j >= take_px
                if stop_hit:
                    exit_idx = j
                    exit_px = float(stop_px)
                    exit_reason = "stop_loss"
                    break
                if take_hit:
                    exit_idx = j
                    exit_px = float(take_px)
                    exit_reason = "take_profit"
                    break
                if bool(exit_sig.iat[j]):
                    exit_idx = j
                    exit_px = float(close_j) if pd.notna(close_j) else float(entry_open)
                    exit_reason = "event_exit"
                    break
                if (j - entry_idx + 1) >= max_hold_bars:
                    exit_idx = j
                    exit_px = float(close_j) if pd.notna(close_j) else float(entry_open)
                    exit_reason = "time_exit"
                    break
                j += 1

            if exit_idx is None:
                if enforce_t1 and last_sellable_idx is None:
                    t1_no_sellbar_skips += 1
                    i = int(entry_idx + 1 + cooldown_bars)
                    continue
                exit_idx = int(last_sellable_idx) if (enforce_t1 and last_sellable_idx is not None) else (n - 1)
                close_last = pd.to_numeric(x.at[exit_idx, "close"], errors="coerce")
                exit_px = float(close_last) if pd.notna(close_last) else float(entry_open)
                exit_reason = "eod_exit"

            entry_exec = float(entry_open) * (1 + fee_cost)
            exit_exec = float(exit_px) * (1 - fee_cost)
            ret = (exit_exec / entry_exec) - 1
            candidates.append(
                {
                    "code": code,
                    "signal_date": x.at[i, "date"],
                    "entry_date": x.at[entry_idx, "date"],
                    "exit_date": x.at[exit_idx, "date"],
                    "entry_signal": signal_tag,
                    "entry_quality_score": float(quality_score),
                    "entry_phase": phase_label,
                    "entry_phase_score": float(phase_score),
                    "entry_events_weight": float(event_weight),
                    "entry_structure_score": int(structure_score),
                    "entry_trend_score": float(trend_component),
                    "entry_volatility_score": float(volume_signal + volatility_penalty),
                    "entry_price": float(entry_open),
                    "exit_price": float(exit_px),
                    "bars_held": int(exit_idx - entry_idx + 1),
                    "exit_reason": exit_reason,
                    "raw_ret_pct": float(ret * 100),
                    "entry_exec": float(entry_exec),
                    "exit_exec": float(exit_exec),
                }
            )
            i = int(exit_idx + 1 + cooldown_bars)

    candidates_df = pd.DataFrame(candidates)
    if enforce_t1 and t1_no_sellbar_skips > 0:
        notes.append(f"T+1约束下有 {t1_no_sellbar_skips} 笔信号因样本内无可卖出日被跳过。")
    if candidates_df.empty:
        return candidates_df, pd.DataFrame(), empty_metrics, notes

    # Step 2: portfolio execution with capital, sizing and max concurrent positions.
    if prioritize_signals:
        if priority_mode == "phase_first":
            sort_cols = ["entry_date", "entry_phase_score", "entry_quality_score", "entry_events_weight", "entry_structure_score", "code", "exit_date"]
        elif priority_mode == "momentum":
            sort_cols = ["entry_date", "entry_trend_score", "entry_quality_score", "entry_events_weight", "entry_structure_score", "code", "exit_date"]
        else:
            sort_cols = ["entry_date", "entry_quality_score", "entry_phase_score", "entry_events_weight", "entry_structure_score", "code", "exit_date"]
        candidates_df = candidates_df.sort_values(
            sort_cols,
            ascending=[True, False, False, False, False, True, True],
        ).reset_index(drop=True)
        notes.append(f"同日信号已按优先分排序执行（模式：{priority_mode_label}）。")
        if priority_topk_per_day > 0:
            before_cnt = int(len(candidates_df))
            candidates_df = (
                candidates_df
                .groupby("entry_date", sort=False, group_keys=False)
                .head(priority_topk_per_day)
                .reset_index(drop=True)
            )
            dropped_cnt = before_cnt - int(len(candidates_df))
            if dropped_cnt > 0:
                notes.append(f"同日TopK限流已生效：每日保留前 {priority_topk_per_day} 笔候选，共过滤 {dropped_cnt} 笔。")
    else:
        candidates_df = candidates_df.sort_values(["entry_date", "code", "exit_date"]).reset_index(drop=True)
        if priority_topk_per_day > 0:
            notes.append("已设置同日TopK限流，但未启用“同日信号优先级排序”，TopK未生效。")
    cash = float(initial_capital)
    equity = float(initial_capital)  # realized equity
    max_concurrent = 0
    active_positions: List[Dict] = []
    executed = []
    skip_reasons = {
        "max_positions": 0,
        "insufficient_cash": 0,
        "invalid_price": 0,
        "risk_mode_unavailable": 0,
    }
    risk_mode_blocked = False
    min_mode_fallback = False

    def _release_until(cur_date: pd.Timestamp) -> None:
        nonlocal cash, equity, active_positions
        if not active_positions:
            return
        remain = []
        # Conservative: only release positions that exited strictly before current entry date.
        for pos in active_positions:
            if pd.to_datetime(pos["exit_date"]) < pd.to_datetime(cur_date):
                cash += float(pos["exit_amount"])
                equity += float(pos["pnl_amount"])
            else:
                remain.append(pos)
        active_positions = remain

    for row in candidates_df.itertuples(index=False):
        entry_date = pd.to_datetime(row.entry_date)
        _release_until(entry_date)

        if len(active_positions) >= max_positions:
            skip_reasons["max_positions"] += 1
            continue

        entry_exec = float(row.entry_exec)
        exit_exec = float(row.exit_exec)
        if (not np.isfinite(entry_exec)) or (not np.isfinite(exit_exec)) or entry_exec <= 0:
            skip_reasons["invalid_price"] += 1
            continue

        fixed_alloc = max(0.0, equity * position_pct)
        risk_alloc = np.nan
        if stop_loss > 0 and risk_per_trade > 0:
            risk_budget = equity * risk_per_trade
            loss_per_1yuan = stop_loss + 2 * fee_cost
            if loss_per_1yuan > 0:
                risk_alloc = risk_budget / loss_per_1yuan

        if position_mode == "fixed":
            alloc = fixed_alloc
        elif position_mode == "risk":
            if np.isfinite(risk_alloc) and float(risk_alloc) > 0:
                alloc = float(risk_alloc)
            else:
                skip_reasons["risk_mode_unavailable"] += 1
                risk_mode_blocked = True
                continue
        else:
            if np.isfinite(risk_alloc) and float(risk_alloc) > 0:
                alloc = min(fixed_alloc, float(risk_alloc))
            else:
                alloc = fixed_alloc
                if risk_per_trade > 0:
                    min_mode_fallback = True

        alloc = min(alloc, cash)
        if alloc < entry_exec:
            skip_reasons["insufficient_cash"] += 1
            continue

        shares = int(np.floor(alloc / entry_exec))
        if shares <= 0:
            skip_reasons["insufficient_cash"] += 1
            continue

        invested = shares * entry_exec
        exit_amount = shares * exit_exec
        pnl_amount = exit_amount - invested
        ret_pct = (pnl_amount / invested) * 100 if invested > 0 else 0.0
        cash -= invested

        active_positions.append(
            {
                "exit_date": pd.to_datetime(row.exit_date),
                "exit_amount": float(exit_amount),
                "pnl_amount": float(pnl_amount),
            }
        )
        max_concurrent = max(max_concurrent, len(active_positions))

        executed.append(
            {
                "code": row.code,
                "signal_date": pd.to_datetime(row.signal_date),
                "entry_date": pd.to_datetime(row.entry_date),
                "exit_date": pd.to_datetime(row.exit_date),
                "entry_signal": row.entry_signal,
                "entry_quality_score": float(getattr(row, "entry_quality_score", 0.0)),
                "entry_phase": str(getattr(row, "entry_phase", "阶段未明")),
                "entry_phase_score": float(getattr(row, "entry_phase_score", 0.0)),
                "entry_events_weight": float(getattr(row, "entry_events_weight", 0.0)),
                "entry_structure_score": int(getattr(row, "entry_structure_score", 0)),
                "entry_trend_score": float(getattr(row, "entry_trend_score", 0.0)),
                "entry_volatility_score": float(getattr(row, "entry_volatility_score", 0.0)),
                "entry_price": float(row.entry_price),
                "exit_price": float(row.exit_price),
                "bars_held": int(row.bars_held),
                "exit_reason": row.exit_reason,
                "shares": int(shares),
                "position_value": float(invested),
                "exit_value": float(exit_amount),
                "pnl_amount": float(pnl_amount),
                "ret_pct": float(ret_pct),
            }
        )

    if active_positions:
        for pos in sorted(active_positions, key=lambda x: pd.to_datetime(x["exit_date"])):
            cash += float(pos["exit_amount"])
            equity += float(pos["pnl_amount"])

    trades_df = pd.DataFrame(executed)
    total_candidates = int(len(candidates_df))
    skipped = int(sum(skip_reasons.values()))
    fill_rate = (len(trades_df) / total_candidates * 100.0) if total_candidates > 0 else 0.0
    if risk_mode_blocked:
        notes.append("风险仓位模式需要有效止损和风险预算；当前参数下部分信号被跳过。")
    if min_mode_fallback and position_mode == "min":
        notes.append("当前为取最小模式，但未启用有效风险预算，已回退为固定仓位。")
    if skipped > 0:
        detail = ", ".join([f"{k}:{v}" for k, v in skip_reasons.items() if v > 0])
        notes.append(f"组合约束跳过 {skipped} 笔信号（{detail}）。")
    if trades_df.empty:
        eq_df = _build_mark_to_market_curve(
            trades_df,
            fallback_start=pd.to_datetime(candidates_df["entry_date"].min()),
        )
        if not eq_df.empty:
            eq_df = eq_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            eq_df["peak"] = eq_df["equity"].cummax()
            eq_df["drawdown"] = (eq_df["equity"] / eq_df["peak"]) - 1
            max_dd = float(eq_df["drawdown"].min() * 100)
            final_eq = float(eq_df["equity"].iloc[-1])
        else:
            max_dd = 0.0
            final_eq = float(equity)
        metrics = empty_metrics.copy()
        metrics.update(
            {
                "final_equity": float(final_eq),
                "skipped_trades": float(skipped),
                "fill_rate_pct": float(fill_rate),
                "max_concurrent_positions": float(max_concurrent),
                "cum_return_pct": float((final_eq / initial_capital - 1) * 100),
                "max_drawdown_pct": float(max_dd),
            }
        )
        return trades_df, eq_df, metrics, notes

    trades_df = trades_df.sort_values(["entry_date", "exit_date", "code"]).reset_index(drop=True)
    r = trades_df["ret_pct"] / 100.0
    win_r = r[r > 0]
    loss_r = r[r < 0]
    win_rate = float((r > 0).mean() * 100)
    avg_ret = float(r.mean() * 100)
    avg_win = float(win_r.mean() * 100) if len(win_r) else 0.0
    avg_loss = float(loss_r.mean() * 100) if len(loss_r) else 0.0
    profit_factor = float(win_r.sum() / abs(loss_r.sum())) if len(loss_r) else (999.0 if len(win_r) else 0.0)
    payoff_ratio = float((win_r.mean() / abs(loss_r.mean()))) if len(loss_r) and len(win_r) else 0.0

    eq_df = _build_mark_to_market_curve(trades_df)
    if not eq_df.empty:
        eq_df = eq_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["drawdown"] = (eq_df["equity"] / eq_df["peak"]) - 1
    max_dd = float(eq_df["drawdown"].min() * 100) if not eq_df.empty else 0.0
    final_equity = float(eq_df["equity"].iloc[-1]) if not eq_df.empty else float(equity)

    metrics = {
        "initial_capital": float(initial_capital),
        "final_equity": float(final_equity),
        "total_trades": float(len(trades_df)),
        "skipped_trades": float(skipped),
        "fill_rate_pct": float(fill_rate),
        "max_concurrent_positions": float(max_concurrent),
        "win_rate_pct": win_rate,
        "avg_ret_pct": avg_ret,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "cum_return_pct": float((final_equity / initial_capital - 1) * 100),
        "max_drawdown_pct": max_dd,
    }
    return trades_df, eq_df, metrics, notes



def main():
    st.set_page_config(page_title="趋势选股系统", layout="wide")
    st.title("趋势选股系统")
    st.caption("TDX/CSV -> 分层筛选 -> 威科夫结构分析 -> 事件回测")
    init_settings_state()
    sanitize_widget_state()

    tab_data, tab_meta, tab_display, tab_analysis, tab_backtest = st.tabs(
        ["📂 数据输入", "📋 元数据", "🎨 显示设置", "📊 分析", "🔬 回测"]
    )

    # ── helpers to get shared state ──
    def _get_encoding() -> str:
        return str(st.session_state.get("csv_encoding", "auto"))

    def _get_price_df() -> Optional[pd.DataFrame]:
        return st.session_state.get("loaded_price_df")

    def _get_meta_df() -> pd.DataFrame:
        return st.session_state.get("current_meta_df", pd.DataFrame())

    def _prepare_price_df(price_df, meta_df) -> Optional[pd.DataFrame]:
        if price_df is None or price_df.empty:
            return None
        price_df = coerce_types(price_df.copy())
        meta_df = coerce_types(meta_df.copy()) if meta_df is not None and not meta_df.empty else pd.DataFrame()
        if meta_df is not None and not meta_df.empty and "code" in meta_df.columns:
            meta_df["code"] = normalize_code(meta_df["code"])
        price_df = merge_meta(price_df, meta_df)
        if "code" not in price_df.columns or "date" not in price_df.columns:
            return None
        price_df["code"] = normalize_code(price_df["code"])
        price_df = price_df[price_df["code"].notna()].copy()
        price_df = ensure_market_board(price_df)
        return price_df

    # ================================================================
    # Tab 1: 数据输入
    # ================================================================
    with tab_data:
        mode = st.radio(
            "数据来源",
            ["TDX本地数据", "CSV单文件(多股票)", "CSV文件夹(每股一文件)"],
            key="data_mode",
        )
        encoding = st.selectbox("CSV编码", ["auto", "utf-8", "gbk", "gb2312"], key="csv_encoding")

        if mode == "TDX本地数据":
            max_days = st.number_input("读取最近N交易日", min_value=80, value=600, step=20, key="tdx_max_days")
            deep_scan = st.checkbox("深度扫描磁盘", value=False, key="tdx_deep_scan")
            quick_found = [Path(p) for p in cached_tdx_paths(False)]
            found = quick_found
            if deep_scan:
                st.caption("深度扫描较慢，不会自动执行。点击按钮后更新候选路径。")
                if st.button("执行深度扫描", key="tdx_run_deep_scan_btn"):
                    with st.spinner("正在深度扫描磁盘，请稍候..."):
                        deep_found = [Path(p) for p in cached_tdx_paths(True)]
                    st.session_state["tdx_deep_found_paths"] = [str(p) for p in deep_found]
                deep_cached = [Path(p) for p in st.session_state.get("tdx_deep_found_paths", [])]
                if deep_cached:
                    found = dedupe_paths(quick_found + deep_cached)
            else:
                st.session_state.pop("tdx_deep_found_paths", None)

            options = [""] + [str(p) for p in found]
            cur_selected = str(st.session_state.get("tdx_vipdoc_selected", "") or "")
            if cur_selected and cur_selected not in options:
                st.session_state.pop("tdx_vipdoc_selected", None)
            selected = st.selectbox("检测到的vipdoc", options, key="tdx_vipdoc_selected")
            manual_path = st.text_input("手动vipdoc路径", value=st.session_state.get("tdx_manual_path", ""), key="tdx_manual_path")
            if st.button("加载TDX数据", key="load_tdx_data_btn"):
                raw_path = (manual_path or selected).strip()
                if not raw_path:
                    st.error("请先选择或输入vipdoc路径。")
                else:
                    vipdoc = resolve_vipdoc_path(Path(raw_path))
                    if vipdoc is None:
                        st.error("vipdoc路径无效，请检查。")
                    else:
                        loaded_df = load_tdx_daily(vipdoc, int(max_days))
                        if loaded_df.empty:
                            st.error("未读取到TDX日线数据。")
                        else:
                            st.session_state["loaded_price_df"] = loaded_df
                            st.success(f"已加载 {loaded_df['code'].nunique()} 只股票，共 {len(loaded_df):,} 行。")

        elif mode == "CSV单文件(多股票)":
            price_file = st.file_uploader("上传行情CSV", type=["csv"], key="price_csv_file")
            if price_file is not None:
                try:
                    st.session_state["loaded_price_df"] = load_price_from_file(price_file, encoding=encoding)
                except Exception as exc:
                    st.error(f"读取行情CSV失败：{exc}")

        else:  # CSV文件夹
            folder_path = st.text_input("CSV文件夹路径", value=st.session_state.get("csv_folder_path", ""), key="csv_folder_path")
            code_regex = st.text_input("文件名提取代码正则(含分组)", value=st.session_state.get("csv_code_regex", r"(\d{6})"), key="csv_code_regex")
            if st.button("读取CSV文件夹", key="load_csv_folder_btn"):
                if not folder_path.strip():
                    st.error("请先输入CSV文件夹路径。")
                else:
                    try:
                        loaded_df = load_price_from_folder(Path(folder_path.strip()), encoding=encoding, code_regex=code_regex)
                        if loaded_df.empty:
                            st.warning("未读取到有效CSV数据，请检查文件内容和正则。")
                        else:
                            st.session_state["loaded_price_df"] = loaded_df
                            st.success(f"已加载 {loaded_df['code'].nunique()} 只股票，共 {len(loaded_df):,} 行。")
                    except Exception as exc:
                        st.error(f"读取CSV文件夹失败：{exc}")

        # Data status
        if "loaded_price_df" in st.session_state:
            pdf = st.session_state["loaded_price_df"]
            st.info(f"✅ 当前已加载数据：{pdf['code'].nunique()} 只股票，{len(pdf):,} 行。")
        else:
            st.warning("尚未加载行情数据，请先加载。")

        persist_settings(keys=[
            "data_mode", "csv_encoding", "tdx_max_days", "tdx_deep_scan",
            "tdx_vipdoc_selected", "tdx_manual_path", "csv_folder_path", "csv_code_regex",
        ])

    # ================================================================
    # Tab 2: 元数据
    # ================================================================
    with tab_meta:
        encoding = _get_encoding()
        meta_df = pd.DataFrame()
        api_meta_df = pd.DataFrame()

        meta_file = st.file_uploader("上传元数据CSV(可选)", type=["csv"], key="meta_csv_file")
        if meta_file is not None:
            try:
                meta_df = load_meta(meta_file, encoding=encoding)
            except Exception as exc:
                st.error(f"读取元数据失败：{exc}")
        if "api_meta_df" in st.session_state and isinstance(st.session_state["api_meta_df"], pd.DataFrame):
            api_meta_df = st.session_state["api_meta_df"]

        st.subheader("元数据API补全")
        meta_api_enable = st.checkbox("启用元数据API补全", value=False, key="meta_api_enable")
        if meta_api_enable:
            meta_api_url = st.text_input("API地址", value=st.session_state.get("meta_api_url", ""), key="meta_api_url")
            meta_api_method = st.selectbox("请求方法", ["GET", "POST"], index=0, key="meta_api_method")
            meta_api_code_param = st.text_input("代码参数名", value=st.session_state.get("meta_api_code_param", "codes"), key="meta_api_code_param")
            meta_api_payload_style = st.selectbox("代码传参格式", ["comma", "array"], index=0, key="meta_api_payload_style")
            meta_api_data_key = st.text_input("数据字段键(可选)", value=st.session_state.get("meta_api_data_key", ""), key="meta_api_data_key")
            meta_api_headers = st.text_area("请求头JSON", value=st.session_state.get("meta_api_headers", "{}"), key="meta_api_headers")
            meta_api_timeout = st.number_input("超时(秒)", min_value=3, max_value=120, value=int(st.session_state.get("meta_api_timeout", 12)), step=1, key="meta_api_timeout")
            meta_api_max_codes = st.number_input("每次最大代码数", min_value=50, max_value=5000, value=int(st.session_state.get("meta_api_max_codes", 500)), step=50, key="meta_api_max_codes")
            price_df_raw = _get_price_df()
            if st.button("调用API补全元数据", key="meta_api_fetch_btn"):
                if price_df_raw is None or price_df_raw.empty or "code" not in price_df_raw.columns:
                    st.error("请先在「数据输入」页加载包含code列的行情数据。")
                else:
                    codes = normalize_code(price_df_raw["code"]).dropna().astype(str).unique().tolist()
                    codes = sorted(codes)[: int(meta_api_max_codes)]
                    headers = parse_headers_json(meta_api_headers)
                    api_df, api_err = fetch_meta_from_api(
                        url=meta_api_url.strip(),
                        method=meta_api_method,
                        codes=codes,
                        code_param=meta_api_code_param.strip() or "codes",
                        headers=headers,
                        timeout=int(meta_api_timeout),
                        data_key=(meta_api_data_key or "").strip(),
                        payload_style=meta_api_payload_style,
                    )
                    if api_err:
                        st.error(api_err)
                    elif api_df is None or api_df.empty:
                        st.warning("API未返回可用元数据。")
                    else:
                        api_df = normalize_columns(api_df)
                        if "code" in api_df.columns:
                            api_df["code"] = normalize_code(api_df["code"])
                        st.session_state["api_meta_df"] = api_df
                        api_meta_df = api_df
                        st.success(f"API元数据已获取：{len(api_df)} 行。")
            if api_meta_df is not None and not api_meta_df.empty:
                st.caption(f"API元数据缓存：{len(api_meta_df)} 行。")

        # Column mapping
        price_mapping_targets = {
            "date": "日期列", "open": "开盘列", "high": "最高列", "low": "最低列",
            "close": "收盘列", "volume": "成交量列", "amount": "成交额列", "code": "代码列",
            "name": "名称列", "market": "市场列", "board": "板块列", "industry": "行业列",
        }
        meta_mapping_targets = {
            "code": "代码列", "name": "名称列", "board": "板块列", "industry": "行业列",
            "list_days": "上市天数列", "list_date": "上市日期列", "float_mv": "流通市值列",
            "market": "市场列", "sector_return_5d": "板块5日涨幅列",
        }

        price_df_raw = _get_price_df()
        if price_df_raw is not None and not price_df_raw.empty:
            price_mapping = column_mapping_ui(price_df_raw, "行情列映射(可选)", price_mapping_targets, "price_map")
            if price_mapping:
                price_df_raw = apply_column_mapping(price_df_raw.copy(), price_mapping)
                st.session_state["loaded_price_df"] = price_df_raw
        if meta_df is not None and not meta_df.empty:
            meta_mapping = column_mapping_ui(meta_df, "元数据列映射(可选)", meta_mapping_targets, "meta_map")
            if meta_mapping:
                meta_df = apply_column_mapping(meta_df.copy(), meta_mapping)
        if meta_api_enable and api_meta_df is not None and not api_meta_df.empty:
            api_mapping = column_mapping_ui(api_meta_df, "API元数据列映射(可选)", meta_mapping_targets, "api_meta_map")
            if api_mapping:
                api_meta_df = apply_column_mapping(api_meta_df.copy(), api_mapping)
            meta_df = merge_meta_with_api(meta_df, api_meta_df)

        # Store merged meta for other tabs
        st.session_state["current_meta_df"] = meta_df

        persist_settings(keys=[
            "meta_api_enable", "meta_api_url", "meta_api_method", "meta_api_code_param",
            "meta_api_payload_style", "meta_api_data_key", "meta_api_headers",
            "meta_api_timeout", "meta_api_max_codes",
        ])

    # ================================================================
    # Tab 3: 显示设置
    # ================================================================
    with tab_display:
        color_up_red = st.checkbox("红涨绿跌", value=True, key="color_up_red")
        use_aggrid = st.checkbox("使用交互表格(AgGrid)", value=True, key="use_aggrid")

        trend_window_preview = int(st.session_state.get("trend_window", 20))
        display_options = make_display_columns(trend_window_preview)
        final_cols_default = make_default_columns(trend_window_preview, "final")
        candidate_cols_default = make_default_columns(trend_window_preview, "candidate")
        if "final_cols" in st.session_state:
            normalized_final = normalize_display_selection(
                st.session_state.get("final_cols", []), trend_window_preview, display_options,
            )
            if normalized_final:
                st.session_state["final_cols"] = normalized_final
            else:
                st.session_state.pop("final_cols", None)
        if "candidate_cols" in st.session_state:
            normalized_candidate = normalize_display_selection(
                st.session_state.get("candidate_cols", []), trend_window_preview, display_options,
            )
            if normalized_candidate:
                st.session_state["candidate_cols"] = normalized_candidate
            else:
                st.session_state.pop("candidate_cols", None)
        final_cols = st.multiselect(
            "最终清单显示列", display_options,
            default=normalize_display_selection(
                st.session_state.get("final_cols", final_cols_default), trend_window_preview, display_options,
            ) or final_cols_default,
            key="final_cols",
        )
        candidate_cols = st.multiselect(
            "候选池显示列", display_options,
            default=normalize_display_selection(
                st.session_state.get("candidate_cols", candidate_cols_default), trend_window_preview, display_options,
            ) or candidate_cols_default,
            key="candidate_cols",
        )

        persist_settings(keys=["color_up_red", "use_aggrid", "final_cols", "candidate_cols"])

    # ================================================================
    # Tab 4: 分析
    # ================================================================
    with tab_analysis:
        price_df_raw = _get_price_df()
        meta_df = _get_meta_df()
        if price_df_raw is None or price_df_raw.empty:
            st.warning("请先在「数据输入」页加载行情数据。")
        else:
            price_df = _prepare_price_df(price_df_raw, meta_df)
            if price_df is None:
                st.error("行情数据缺少 code 或 date 列，请检查数据。")
            else:
                params_col, results_col = st.columns([1, 2], gap="large")

                with params_col:
                    st.markdown("### 分析参数")
                    with st.form("analysis_form"):
                        eval_default = dt.date.today()
                        if "date" in price_df.columns:
                            max_dt = pd.to_datetime(price_df["date"], errors="coerce").max()
                            if pd.notna(max_dt):
                                eval_default = max_dt.date()
                        eval_date = st.date_input("评估日期", value=eval_default, key="eval_date")

                        st.markdown("**基础过滤**")
                        only_stocks = st.checkbox("仅A股股票", value=True, key="only_stocks")
                        enable_market = st.checkbox("市场过滤", value=True, key="enable_market")
                        market_label_map = {"沪市(sh)": "sh", "深市(sz)": "sz", "北交所(bj)": "bj"}
                        markets_selected = st.multiselect(
                            "交易所", list(market_label_map.keys()),
                            default=st.session_state.get("markets_selected", list(market_label_map.keys())),
                            key="markets_selected",
                        )
                        markets = [market_label_map[m] for m in markets_selected]
                        enable_board = st.checkbox("板块过滤", value=True, key="enable_board")
                        boards = st.multiselect(
                            "板块", ["主板", "创业板", "科创板", "北交所"],
                            default=st.session_state.get("boards", ["主板", "创业板", "科创板", "北交所"]),
                            key="boards",
                        )
                        enable_st = st.checkbox("剔除ST", value=True, key="enable_st")
                        enable_list_days = st.checkbox("上市天数过滤", value=True, key="enable_list_days")
                        min_list_days = st.number_input("最少上市天数", min_value=0, value=250, step=10, key="min_list_days")
                        enable_float_mv = st.checkbox("流通市值过滤", value=True, key="enable_float_mv")
                        float_mv_unit = st.selectbox("流通市值单位", ["亿", "万元", "元"], index=0, key="float_mv_unit")
                        float_mv_threshold = st.number_input("最小流通市值", value=30.0, step=1.0, key="float_mv_threshold")

                        st.markdown("**趋势参数**")
                        trend_window = st.number_input("趋势窗口(交易日)", min_value=5, max_value=120, value=20, step=5, key="trend_window")
                        ret_min = st.number_input(f"{trend_window}日涨幅下限(%)", value=20.0, step=1.0, key="ret_min") / 100
                        ret_max = st.number_input(f"{trend_window}日涨幅上限(%)", value=100.0, step=5.0, key="ret_max") / 100
                        top_n = st.number_input("涨幅Top N", min_value=50, value=300, step=50, key="top_n")
                        trend_min_conditions = st.number_input("趋势条件最少满足", min_value=1, max_value=4, value=3, key="trend_min_conditions")
                        up_down_ratio = st.number_input("量价配合阈值", value=1.2, step=0.1, key="up_down_ratio")
                        dd_min = st.number_input("回撤下限(%)", value=5.0, step=1.0, key="dd_min") / 100
                        dd_max = st.number_input("回撤上限(%)", value=25.0, step=1.0, key="dd_max") / 100
                        drawdown_mode = st.selectbox("回撤口径", ["当前回撤", "最大回撤"], index=0, key="drawdown_mode")

                        st.markdown("**近期拐头向上**")
                        enable_turn_up = st.checkbox("启用拐头向上过滤", value=False, key="enable_turn_up")
                        st.caption("说明：MA5/10/20 是均线周期；后面的 3/5/10 日是斜率回看窗口。")
                        turn_up_ma5 = st.number_input("MA5 3日斜率下限(%)", value=0.0, step=0.1, key="turn_up_ma5") / 100
                        turn_up_ma10 = st.number_input("MA10 5日斜率下限(%)", value=0.0, step=0.1, key="turn_up_ma10") / 100
                        turn_up_ma20 = st.number_input("MA20 10日斜率下限(%)", value=0.0, step=0.1, key="turn_up_ma20") / 100

                        st.markdown("**威科夫事件过滤**")
                        st.caption("仅影响分层筛选结果（Layer1起生效）。")
                        enable_wyckoff_event_filter = st.checkbox("启用威科夫事件过滤", value=False, key="enable_wyckoff_event_filter")
                        wy_event_options = WYCKOFF_EVENT_OPTIONS
                        wy_required_events = st.multiselect(
                            "必须包含事件", wy_event_options,
                            default=st.session_state.get("wy_required_events", []),
                            format_func=format_wyckoff_event_label, key="wy_required_events",
                        )
                        wy_require_sequence = st.checkbox(
                            "要求标准8步序列完整(PS->SC->AR->ST->Spring->SOS->JOC->LPS)",
                            value=False, key="wy_require_sequence",
                        )
                        wy_event_lookback = st.number_input("事件回看窗口(交易日)", min_value=30, max_value=300, value=120, step=10, key="wy_event_lookback")

                        st.markdown("**威科夫阶段池**")
                        wy_phase_scope = st.selectbox(
                            "阶段筛选股票池来源",
                            ["All symbols", "Layer1", "Layer4 candidates", "Final Top"],
                            index=0,
                            format_func=lambda x: {
                                "All symbols": "全市场", "Layer1": "第1层（基础过滤后）",
                                "Layer4 candidates": "第4层候选池", "Final Top": "最终Top",
                            }.get(x, x),
                            key="wy_phase_scope",
                        )
                        wy_phase_selected = st.multiselect(
                            "目标阶段（留空=不过滤）", WYCKOFF_PHASE_OPTIONS,
                            default=st.session_state.get("wy_phase_selected", []), key="wy_phase_selected",
                        )
                        wy_phase_events = st.multiselect(
                            "阶段池要求事件（可选）", wy_event_options,
                            default=st.session_state.get("wy_phase_events", []),
                            format_func=format_wyckoff_event_label, key="wy_phase_events",
                        )

                        st.markdown("**买点与评分**")
                        buy_min = st.number_input("买点距离下限(%)", value=-5.0, step=1.0, key="buy_min") / 100
                        buy_max = st.number_input("买点距离上限(%)", value=10.0, step=1.0, key="buy_max") / 100
                        w_sector = st.slider("题材强度权重", 0.0, 1.0, 0.4, 0.05, key="w_sector")
                        w_dd = st.slider("回撤适中权重", 0.0, 1.0, 0.25, 0.05, key="w_dd")
                        w_amount = st.slider("成交额权重", 0.0, 1.0, 0.2, 0.05, key="w_amount")
                        w_vol = st.slider("波动率权重", 0.0, 1.0, 0.15, 0.05, key="w_vol")
                        amount_min_yi = st.number_input("日均成交额阈值(亿)", value=5.0, step=1.0, key="amount_min_yi")
                        final_n = st.number_input("最终保留数量", min_value=1, max_value=20, value=5, key="final_n")

                        submit_analysis = st.form_submit_button("🚀 开始分析")

                    if submit_analysis:
                        st.session_state["analysis_triggered"] = True

                    persist_settings(keys=[
                        "eval_date", "only_stocks", "enable_market", "markets_selected", "enable_board", "boards",
                        "enable_st", "enable_list_days", "min_list_days", "enable_float_mv", "float_mv_unit",
                        "float_mv_threshold", "ret_min", "ret_max", "top_n", "trend_window", "trend_min_conditions",
                        "up_down_ratio", "dd_min", "dd_max", "drawdown_mode", "enable_turn_up", "turn_up_ma5",
                        "turn_up_ma10", "turn_up_ma20", "enable_wyckoff_event_filter", "wy_required_events",
                        "wy_require_sequence", "wy_event_lookback", "wy_phase_scope", "wy_phase_selected",
                        "wy_phase_events", "buy_min", "buy_max", "w_sector", "w_dd", "w_amount", "w_vol",
                        "amount_min_yi", "final_n",
                    ])

                # ── Analysis results (right column) ──
                with results_col:
                    if only_stocks:
                        price_df = filter_only_stocks(price_df)

                    if not st.session_state.get("analysis_triggered"):
                        st.info("请在左侧配置参数后点击「开始分析」。")
                    else:
                        # Run or use cached analysis
                        if submit_analysis or "analysis_cache" not in st.session_state:
                            df = price_df.copy()
                            if eval_date is not None:
                                df = df[df["date"] <= pd.to_datetime(eval_date)]

                            warnings: List[str] = []
                            if "amount" not in df.columns and "volume" in df.columns:
                                df["amount"] = df["close"] * df["volume"]
                                warnings.append("缺少amount字段，已使用 close*volume 估算。")
                            if "volume" not in df.columns and "amount" not in df.columns:
                                warnings.append("缺少volume/amount字段，量价判断可能受影响。")

                            df = add_features(df, trend_window=int(trend_window))
                            latest = df.groupby("code").tail(1).copy()
                            latest = enrich_wyckoff_latest(latest, win=60)

                            seq_df = compute_wyckoff_sequence_features(df, win=60, lookback=int(wy_event_lookback))
                            if seq_df is not None and not seq_df.empty:
                                latest = latest.merge(seq_df, on="code", how="left")
                            for col_name in ("wy_event_count", "wy_events_present", "wy_sequence_ok"):
                                if col_name not in latest.columns:
                                    latest[col_name] = 0 if col_name == "wy_event_count" else ("无" if col_name == "wy_events_present" else False)

                            float_mv_threshold_value = float_mv_threshold
                            if float_mv_unit == "亿":
                                float_mv_threshold_value = float_mv_threshold * 1e8
                            elif float_mv_unit == "万元":
                                float_mv_threshold_value = float_mv_threshold * 1e4

                            params = {
                                "enable_market": enable_market, "markets": markets,
                                "enable_board": enable_board, "boards": boards,
                                "enable_st": enable_st, "enable_list_days": enable_list_days,
                                "min_list_days": min_list_days, "enable_float_mv": enable_float_mv,
                                "min_float_mv": float_mv_threshold_value,
                                "ret_min": ret_min, "ret_max": ret_max, "top_n": int(top_n),
                                "trend_window": int(trend_window),
                                "trend_min_conditions": int(trend_min_conditions),
                                "up_down_ratio": up_down_ratio, "dd_min": dd_min, "dd_max": dd_max,
                                "drawdown_mode": drawdown_mode,
                                "enable_turn_up": enable_turn_up,
                                "turn_up_ma5": turn_up_ma5, "turn_up_ma10": turn_up_ma10, "turn_up_ma20": turn_up_ma20,
                                "enable_wyckoff_event_filter": enable_wyckoff_event_filter,
                                "wy_required_events": wy_required_events,
                                "wy_require_sequence": wy_require_sequence,
                                "wyckoff_event_lookback": int(wy_event_lookback),
                                "wy_phase_scope": str(wy_phase_scope),
                                "wy_phase_selected": wy_phase_selected,
                                "wy_phase_events": wy_phase_events,
                                "wyckoff_win": 60,
                                "buy_min": buy_min, "buy_max": buy_max,
                                "w_sector": w_sector, "w_dd": w_dd, "w_amount": w_amount, "w_vol": w_vol,
                                "amount_min": amount_min_yi * 1e8, "final_n": int(final_n),
                                "eval_date": pd.to_datetime(eval_date) if eval_date else None,
                            }

                            layers = apply_layer_filters(latest, params, warnings)
                            st.session_state["analysis_cache"] = {
                                "df": df, "latest": latest, "layers": layers,
                                "warnings": warnings, "params": params,
                            }

                        cache = st.session_state.get("analysis_cache", {})
                        df = cache.get("df")
                        latest = cache.get("latest")
                        layers = cache.get("layers")
                        warnings = cache.get("warnings", [])
                        params = cache.get("params", {})
                        color_up_red = st.session_state.get("color_up_red", True)
                        use_aggrid = st.session_state.get("use_aggrid", True)
                        final_cols = st.session_state.get("final_cols", [])
                        candidate_cols = st.session_state.get("candidate_cols", [])
                        win = int(params.get("trend_window", 20))

                        if layers is None:
                            st.warning("分析缓存异常，请重新点击「开始分析」。")
                        else:
                            # Overview metrics
                            st.markdown("### 筛选结果概览")
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("第1层", len(layers["layer1"]))
                            c2.metric("第2层", len(layers["layer2"]))
                            c3.metric("第3层", len(layers["layer3"]))
                            c4.metric("第4层", len(layers["layer4"]))
                            c5.metric("最终Top", len(layers["layer5"]))

                            if warnings:
                                st.warning("\n".join(warnings))

                            tips = build_suggestions(layers, params, warnings, price_df, _get_meta_df())
                            if tips:
                                st.info("\n".join([f"- {t}" for t in tips]))

                            # Final list
                            st.markdown("### 最终待买清单")
                            final_df = format_summary(layers["layer5"], win)
                            final_df = filter_display_columns(final_df, final_cols)
                            render_table(final_df, use_aggrid=use_aggrid, height=420, up_red=color_up_red, trend_window=win)
                            st.download_button(
                                "下载最终清单CSV",
                                data=final_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name="trend_final_list.csv", mime="text/csv",
                                use_container_width=True,
                            )

                            # Candidate pool
                            st.markdown("### 候选池")
                            candidate_df = format_summary(layers["layer4"], win)
                            candidate_df = filter_display_columns(candidate_df, candidate_cols)
                            render_table(candidate_df, use_aggrid=use_aggrid, height=360, up_red=color_up_red, trend_window=win)
                            st.download_button(
                                "下载候选池CSV",
                                data=candidate_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name="trend_candidate_list.csv", mime="text/csv",
                                use_container_width=True,
                            )

                            # Wyckoff phase pool
                            wy_phase_pool_df = build_wyckoff_phase_pool(
                                latest=latest, layers=layers,
                                phase_scope=str(params.get("wy_phase_scope", "All symbols")),
                                phases=params.get("wy_phase_selected", []) or [],
                                required_events=params.get("wy_phase_events", []) or [],
                            )
                            st.markdown("### 威科夫阶段股票池")
                            if wy_phase_pool_df is None or wy_phase_pool_df.empty:
                                st.info("当前威科夫阶段条件下无结果。可放宽阶段或事件条件。")
                            else:
                                phase_count_df = (
                                    wy_phase_pool_df["wyckoff_phase"].fillna("阶段未明")
                                    .value_counts().rename_axis("phase").reset_index(name="count")
                                )
                                st.dataframe(phase_count_df, use_container_width=True, height=220)
                                wy_pool_view = format_summary(wy_phase_pool_df, win)
                                wy_pool_view = filter_display_columns(wy_pool_view, candidate_cols)
                                render_table(wy_pool_view, use_aggrid=use_aggrid, height=320, up_red=color_up_red, trend_window=win)
                                st.download_button(
                                    "下载威科夫阶段池CSV",
                                    data=wy_pool_view.to_csv(index=False).encode("utf-8-sig"),
                                    file_name="wyckoff_phase_pool.csv", mime="text/csv",
                                    use_container_width=True,
                                )

                            # Store for backtest tab
                            st.session_state["wy_phase_pool_df"] = wy_phase_pool_df

                            # K-line chart
                            st.markdown("### 个股走势")
                            chart_options = layers["layer5"]["code"].tolist() or layers["layer4"]["code"].tolist()
                            if chart_options:
                                name_map = {}
                                if "name" in price_df.columns:
                                    name_map = price_df.dropna(subset=["name"]).drop_duplicates("code").set_index("code")["name"].to_dict()
                                code = st.selectbox("选择股票", chart_options, format_func=lambda c: f"{c} {name_map.get(c, '')}".strip(), key="analysis_chart_code")
                                if latest is not None and not latest.empty:
                                    sel = latest[latest["code"] == code].tail(1)
                                    if not sel.empty:
                                        row = sel.iloc[0]
                                        phase = str(row.get("wyckoff_phase", "阶段未明"))
                                        structure = str(row.get("structure_hhh", "-"))
                                        signal = str(row.get("wyckoff_signal", "无"))
                                        ci1, ci2, ci3 = st.columns([2, 1, 1])
                                        ci1.metric("威科夫阶段", phase)
                                        ci2.metric("结构(HH/HL/HC)", structure)
                                        ci3.metric("关键信号", signal)
                                        st.caption(WYCKOFF_PHASE_HINTS.get(phase, WYCKOFF_PHASE_HINTS["阶段未明"]))
                                fig = plot_kline(df, code, up_red=color_up_red)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("暂无可视化股票，请调整筛选参数。")

    # ================================================================
    # Tab 5: 回测
    # ================================================================
    with tab_backtest:
        price_df_raw = _get_price_df()
        meta_df = _get_meta_df()
        if price_df_raw is None or price_df_raw.empty:
            st.warning("请先在「数据输入」页加载行情数据。")
        else:
            price_df = _prepare_price_df(price_df_raw, meta_df)
            if price_df is None:
                st.error("行情数据缺少 code 或 date 列，请检查数据。")
            else:
                bt_params_col, bt_results_col = st.columns([1, 2], gap="large")

                with bt_params_col:
                    st.markdown("### 回测参数")
                    with st.form("backtest_form"):
                        eval_date_bt = st.session_state.get("eval_date", dt.date.today())
                        if not isinstance(eval_date_bt, dt.date):
                            eval_date_bt = dt.date.today()

                        bt_pool = st.selectbox(
                            "回测股票池来源",
                            ["All symbols", "Layer1", "Layer4 candidates", "Final Top", "Wyckoff Phase Pool"],
                            index=2,
                            format_func=lambda x: {
                                "All symbols": "全市场（不依赖筛选）",
                                "Layer1": "第1层（基础过滤后）",
                                "Layer4 candidates": "第4层候选池（默认）",
                                "Final Top": "最终Top（筛选结果最严）",
                                "Wyckoff Phase Pool": "威科夫阶段池（独立策略）",
                            }.get(x, x),
                            key="bt_pool",
                        )
                        bt_range_mode = st.selectbox(
                            "回测区间模式", ["lookback_bars", "custom_dates"], index=0,
                            format_func=lambda x: {"lookback_bars": "按最近K线数", "custom_dates": "自定义日期区间"}.get(x, x),
                            key="bt_range_mode",
                        )
                        bt_lookback_bars = st.number_input(
                            "回测K线数(每股)", min_value=120, max_value=10000,
                            value=int(st.session_state.get("bt_lookback_bars", 1200) or 1200),
                            step=60, key="bt_lookback_bars",
                        )

                        bt_min_date = eval_date_bt
                        bt_max_date = eval_date_bt
                        if "date" in price_df.columns:
                            bt_dates = pd.to_datetime(price_df["date"], errors="coerce").dropna()
                            if not bt_dates.empty:
                                bt_min_date = bt_dates.min().date()
                                bt_max_date = bt_dates.max().date()
                        bt_max_date = min(bt_max_date, eval_date_bt)
                        if bt_max_date < bt_min_date:
                            bt_max_date = bt_min_date

                        bt_start_default = st.session_state.get("bt_start_date")
                        bt_end_default = st.session_state.get("bt_end_date")
                        if not isinstance(bt_start_default, dt.date):
                            bt_start_default = bt_min_date
                        if not isinstance(bt_end_default, dt.date):
                            bt_end_default = bt_max_date
                        bt_start_default = min(max(bt_start_default, bt_min_date), bt_max_date)
                        bt_end_default = min(max(bt_end_default, bt_min_date), bt_max_date)

                        bt_start_date = st.date_input(
                            "回测开始日期", value=bt_start_default,
                            min_value=bt_min_date, max_value=bt_max_date, key="bt_start_date",
                        )
                        bt_end_date = st.date_input(
                            "回测结束日期", value=bt_end_default,
                            min_value=bt_min_date, max_value=bt_max_date, key="bt_end_date",
                        )
                        st.caption("按最近K线数时只使用K线数；自定义日期区间时只使用开始/结束日期。")

                        wy_event_options = WYCKOFF_EVENT_OPTIONS
                        bt_entry_events = st.multiselect(
                            "入场事件", wy_event_options, default=["Spring", "SOS", "JOC", "LPS"],
                            format_func=format_wyckoff_event_label, key="bt_entry_events",
                        )
                        bt_exit_events = st.multiselect(
                            "离场事件", wy_event_options, default=["UTAD", "SOW", "LPSY"],
                            format_func=format_wyckoff_event_label, key="bt_exit_events",
                        )
                        bt_stop_loss = st.number_input("止损(%)", min_value=0.0, max_value=30.0, value=3.0, step=0.5, key="bt_stop_loss") / 100
                        bt_take_profit = st.number_input("止盈(%)", min_value=0.0, max_value=80.0, value=8.0, step=0.5, key="bt_take_profit") / 100
                        bt_max_hold_bars = st.number_input("最大持仓K线", min_value=2, max_value=500, value=30, step=1, key="bt_max_hold_bars")
                        bt_cooldown_bars = st.number_input("冷却K线", min_value=0, max_value=100, value=2, step=1, key="bt_cooldown_bars")
                        bt_fee_bps = st.number_input("单边交易成本(bps)", min_value=0.0, max_value=200.0, value=8.0, step=1.0, key="bt_fee_bps")
                        bt_initial_capital = st.number_input("初始资金", min_value=10000.0, value=1000000.0, step=10000.0, key="bt_initial_capital")
                        bt_position_pct = st.number_input("单笔目标仓位(%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, key="bt_position_pct") / 100
                        bt_risk_per_trade = st.number_input("单笔风险预算(%)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, key="bt_risk_per_trade") / 100
                        bt_position_mode = st.selectbox(
                            "仓位模式", ["min", "fixed", "risk"], index=0,
                            format_func=lambda m: {"min": "取最小(固定∩风险)", "fixed": "固定仓位", "risk": "风险仓位"}.get(m, m),
                            key="bt_position_mode",
                        )
                        bt_prioritize_signals = st.checkbox("同日信号优先级排序（优中选优）", value=True, key="bt_prioritize_signals")
                        bt_priority_mode_options = list(BACKTEST_PRIORITY_MODE_LABELS.keys())
                        bt_priority_mode_default = str(st.session_state.get("bt_priority_mode", "balanced"))
                        if bt_priority_mode_default not in bt_priority_mode_options:
                            bt_priority_mode_default = "balanced"
                        bt_priority_mode = st.selectbox(
                            "优中选优模式", bt_priority_mode_options,
                            index=bt_priority_mode_options.index(bt_priority_mode_default),
                            format_func=lambda m: BACKTEST_PRIORITY_MODE_LABELS.get(str(m), str(m)),
                            key="bt_priority_mode", disabled=not bool(bt_prioritize_signals),
                        )
                        bt_priority_topk_per_day = st.number_input(
                            "同日候选TopK(0=不限)", min_value=0, max_value=200,
                            value=int(st.session_state.get("bt_priority_topk_per_day", 0) or 0),
                            step=1, key="bt_priority_topk_per_day", disabled=not bool(bt_prioritize_signals),
                        )
                        st.caption("阶段优先=左侧早期介入，动量优先=右侧强势延续，均衡=通用回测。")
                        bt_enforce_t1 = st.checkbox("A股T+1约束", value=True, key="bt_enforce_t1")
                        bt_max_positions = st.number_input("最大并发持仓", min_value=1, max_value=100, value=5, step=1, key="bt_max_positions")

                        submit_backtest = st.form_submit_button("🚀 开始回测")

                    if submit_backtest:
                        st.session_state["backtest_triggered"] = True

                    persist_settings(keys=[
                        "bt_pool", "bt_range_mode", "bt_start_date", "bt_end_date", "bt_lookback_bars",
                        "bt_entry_events", "bt_exit_events", "bt_stop_loss", "bt_take_profit",
                        "bt_max_hold_bars", "bt_cooldown_bars", "bt_fee_bps", "bt_initial_capital",
                        "bt_position_pct", "bt_risk_per_trade", "bt_position_mode", "bt_prioritize_signals",
                        "bt_priority_mode", "bt_priority_topk_per_day", "bt_enforce_t1", "bt_max_positions",
                    ])

                # ── Backtest results (right column) ──
                with bt_results_col:
                    if not st.session_state.get("backtest_triggered"):
                        st.info("请在左侧配置回测参数后点击「开始回测」。")
                    else:
                        # Prepare data for backtest
                        only_stocks_bt = st.session_state.get("only_stocks", True)
                        if only_stocks_bt:
                            price_df = filter_only_stocks(price_df)

                        trend_window_bt = int(st.session_state.get("trend_window", 20))

                        if submit_backtest or "backtest_cache" not in st.session_state:
                            bt_df = price_df.copy()
                            if eval_date_bt is not None:
                                bt_df = bt_df[bt_df["date"] <= pd.to_datetime(eval_date_bt)]

                            bt_warnings: List[str] = []
                            if "amount" not in bt_df.columns and "volume" in bt_df.columns:
                                bt_df["amount"] = bt_df["close"] * bt_df["volume"]
                            bt_df = add_features(bt_df, trend_window=trend_window_bt)

                            # Enrich wyckoff for backtest
                            bt_latest = bt_df.groupby("code").tail(1).copy()
                            bt_latest = enrich_wyckoff_latest(bt_latest, win=60)

                            # Determine pool codes
                            pool_name = str(bt_pool)
                            analysis_cache = st.session_state.get("analysis_cache", {})
                            layers = analysis_cache.get("layers")

                            if pool_name == "All symbols":
                                pool_codes = sorted(bt_df["code"].dropna().astype(str).unique().tolist()) if "code" in bt_df.columns else []
                            elif pool_name == "Layer1" and layers and "layer1" in layers:
                                pool_codes = sorted(layers["layer1"]["code"].dropna().astype(str).unique().tolist()) if "code" in layers["layer1"].columns else []
                            elif pool_name == "Final Top" and layers and "layer5" in layers:
                                pool_codes = sorted(layers["layer5"]["code"].dropna().astype(str).unique().tolist()) if "code" in layers["layer5"].columns else []
                            elif pool_name == "Wyckoff Phase Pool":
                                wy_pool = st.session_state.get("wy_phase_pool_df", pd.DataFrame())
                                pool_codes = sorted(wy_pool["code"].dropna().astype(str).unique().tolist()) if wy_pool is not None and not wy_pool.empty and "code" in wy_pool.columns else []
                            elif layers and "layer4" in layers:
                                pool_codes = sorted(layers["layer4"]["code"].dropna().astype(str).unique().tolist()) if "code" in layers["layer4"].columns else []
                            else:
                                pool_codes = sorted(bt_df["code"].dropna().astype(str).unique().tolist()) if "code" in bt_df.columns else []
                                if pool_name not in ("All symbols",):
                                    bt_warnings.append(f"股票池「{pool_name}」需要先运行分析，已回退到全市场。")

                            bt_range_mode_value = str(bt_range_mode or "lookback_bars")
                            bt_start_ts = None
                            bt_end_ts = None
                            if bt_range_mode_value == "custom_dates":
                                s_ts = pd.to_datetime(bt_start_date, errors="coerce")
                                e_ts = pd.to_datetime(bt_end_date, errors="coerce")
                                if pd.notna(s_ts):
                                    bt_start_ts = s_ts
                                if pd.notna(e_ts):
                                    bt_end_ts = e_ts
                                if bt_start_ts and bt_end_ts and bt_start_ts > bt_end_ts:
                                    bt_start_ts, bt_end_ts = bt_end_ts, bt_start_ts

                            bt_params = {
                                "bt_pool": bt_pool,
                                "bt_range_mode": bt_range_mode_value,
                                "bt_start_date": bt_start_ts,
                                "bt_end_date": bt_end_ts,
                                "bt_lookback_bars": int(bt_lookback_bars),
                                "bt_entry_events": bt_entry_events,
                                "bt_exit_events": bt_exit_events,
                                "bt_stop_loss": float(bt_stop_loss),
                                "bt_take_profit": float(bt_take_profit),
                                "bt_max_hold_bars": int(bt_max_hold_bars),
                                "bt_cooldown_bars": int(bt_cooldown_bars),
                                "bt_fee_bps": float(bt_fee_bps),
                                "bt_initial_capital": float(bt_initial_capital),
                                "bt_position_pct": float(bt_position_pct),
                                "bt_risk_per_trade": float(bt_risk_per_trade),
                                "bt_position_mode": str(bt_position_mode),
                                "bt_prioritize_signals": bool(bt_prioritize_signals),
                                "bt_priority_mode": str(bt_priority_mode),
                                "bt_priority_topk_per_day": int(bt_priority_topk_per_day),
                                "bt_enforce_t1": bool(bt_enforce_t1),
                                "bt_max_positions": int(bt_max_positions),
                            }

                            trades_df, eq_df, bt_metrics, bt_notes = run_wyckoff_backtest(
                                df=bt_df,
                                entry_events=bt_entry_events,
                                exit_events=bt_exit_events,
                                stop_loss=float(bt_stop_loss),
                                take_profit=float(bt_take_profit),
                                max_hold_bars=int(bt_max_hold_bars),
                                lookback_bars=int(bt_lookback_bars),
                                range_mode=bt_range_mode_value,
                                start_date=bt_start_ts,
                                end_date=bt_end_ts,
                                fee_bps=float(bt_fee_bps),
                                cooldown_bars=int(bt_cooldown_bars),
                                code_pool=pool_codes,
                                win=60,
                                initial_capital=float(bt_initial_capital),
                                position_pct=float(bt_position_pct),
                                risk_per_trade=float(bt_risk_per_trade),
                                max_positions=int(bt_max_positions),
                                position_mode=str(bt_position_mode),
                                prioritize_signals=bool(bt_prioritize_signals),
                                enforce_t1=bool(bt_enforce_t1),
                                priority_mode=str(bt_priority_mode),
                                priority_topk_per_day=int(bt_priority_topk_per_day),
                            )
                            bt_notes = (bt_warnings or []) + (bt_notes or [])

                            st.session_state["backtest_cache"] = {
                                "trades_df": trades_df, "eq_df": eq_df,
                                "bt_metrics": bt_metrics, "bt_notes": bt_notes,
                                "bt_params": bt_params,
                            }

                        bt_cache = st.session_state.get("backtest_cache", {})
                        trades_df = bt_cache.get("trades_df", pd.DataFrame())
                        eq_df = bt_cache.get("eq_df", pd.DataFrame())
                        bt_metrics = bt_cache.get("bt_metrics", {})
                        bt_notes = bt_cache.get("bt_notes", [])
                        bt_params = bt_cache.get("bt_params", {})

                        if bt_notes:
                            st.warning("\n".join(bt_notes))

                        # Metrics display
                        st.markdown("### 回测指标")
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Trades", int(bt_metrics.get("total_trades", 0)))
                        m2.metric("Win Rate", f"{bt_metrics.get('win_rate_pct', 0):.2f}%")
                        m3.metric("Avg Trade", f"{bt_metrics.get('avg_ret_pct', 0):.2f}%")
                        m4.metric("Cum Return", f"{bt_metrics.get('cum_return_pct', 0):.2f}%")
                        m5.metric("Max DD", f"{bt_metrics.get('max_drawdown_pct', 0):.2f}%")
                        n1, n2, n3, n4 = st.columns(4)
                        n1.metric("Initial Equity", f"{bt_metrics.get('initial_capital', 0):,.2f}")
                        n2.metric("Final Equity", f"{bt_metrics.get('final_equity', 0):,.2f}")
                        n3.metric("Fill Rate", f"{bt_metrics.get('fill_rate_pct', 0):.2f}%")
                        n4.metric("Skipped/MaxPos", f"{int(bt_metrics.get('skipped_trades', 0))} / {int(bt_metrics.get('max_concurrent_positions', 0))}")

                        mode_text = {"min": "取最小(固定∩风险)", "fixed": "固定仓位", "risk": "风险仓位"}.get(
                            str(bt_params.get("bt_position_mode", "min")), "取最小(固定∩风险)")
                        priority_text = "开启" if bt_params.get("bt_prioritize_signals", True) else "关闭"
                        priority_mode_text = BACKTEST_PRIORITY_MODE_LABELS.get(
                            str(bt_params.get("bt_priority_mode", "balanced")),
                            str(bt_params.get("bt_priority_mode", "balanced")))
                        topk_val = int(bt_params.get("bt_priority_topk_per_day", 0))
                        topk_text = "不限" if topk_val <= 0 else f"Top{topk_val}/日"
                        t1_text = "开启" if bt_params.get("bt_enforce_t1", True) else "关闭"
                        range_mode_text = str(bt_params.get("bt_range_mode", "lookback_bars"))
                        if range_mode_text == "custom_dates":
                            s_ts = pd.to_datetime(bt_params.get("bt_start_date"), errors="coerce")
                            e_ts = pd.to_datetime(bt_params.get("bt_end_date"), errors="coerce")
                            if pd.notna(s_ts) and pd.notna(e_ts):
                                bt_range_text = f"{s_ts.date()} ~ {e_ts.date()}"
                            elif pd.notna(s_ts):
                                bt_range_text = f">= {s_ts.date()}"
                            elif pd.notna(e_ts):
                                bt_range_text = f"<= {e_ts.date()}"
                            else:
                                bt_range_text = "自定义日期区间"
                        else:
                            bt_range_text = f"最近{int(bt_params.get('bt_lookback_bars', 1200))}根K线/每股"
                        st.caption(
                            f"仓位模式：{mode_text}；同日优先级：{priority_text}；优先模式：{priority_mode_text}；"
                            f"同日限流：{topk_text}；T+1：{t1_text}；回测区间：{bt_range_text}；权益曲线为逐bar盯市(MTM)。"
                        )

                        # Equity curve
                        if eq_df is not None and not eq_df.empty:
                            eq_plot = eq_df.copy()
                            init_cap = max(1.0, float(bt_metrics.get("initial_capital", 1_000_000.0)))
                            eq_plot["cum_ret_pct"] = (eq_plot["equity"] / init_cap - 1) * 100
                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(
                                x=eq_plot["date"], y=eq_plot["cum_ret_pct"],
                                mode="lines", line=dict(color="#1565c0", width=2), name="MTM Equity",
                            ))
                            if "realized_equity" in eq_plot.columns:
                                eq_plot["realized_ret_pct"] = (eq_plot["realized_equity"] / init_cap - 1) * 100
                                fig_eq.add_trace(go.Scatter(
                                    x=eq_plot["date"], y=eq_plot["realized_ret_pct"],
                                    mode="lines", line=dict(color="#616161", width=1.3, dash="dot"), name="Realized",
                                ))
                            fig_eq.update_layout(
                                height=300, margin=dict(l=20, r=20, t=20, b=20),
                                xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                            )
                            st.plotly_chart(fig_eq, use_container_width=True)

                        # Trades table
                        st.markdown("### 交易明细")
                        trades_view = format_backtest_trades_table(trades_df)
                        if trades_df is None or trades_df.empty:
                            st.info("当前回测参数下没有成交记录。")
                        else:
                            st.dataframe(trades_view, use_container_width=True, height=320)

                        d1, d2 = st.columns(2)
                        with d1:
                            st.download_button(
                                "下载回测交易CSV",
                                data=trades_view.to_csv(index=False).encode("utf-8-sig"),
                                file_name="wyckoff_backtest_trades.csv", mime="text/csv",
                                key="download_wyckoff_backtest_csv", use_container_width=True,
                            )
                        with d2:
                            analysis_params = st.session_state.get("analysis_cache", {}).get("params", {})
                            merged_params = {**analysis_params, **bt_params}
                            report_snapshot = collect_sidebar_settings_snapshot(merged_params)
                            report_bytes = build_backtest_report_zip(
                                trades_view=trades_view,
                                eq_df=eq_df if eq_df is not None else pd.DataFrame(),
                                bt_metrics=bt_metrics, bt_notes=bt_notes,
                                params=merged_params, sidebar_snapshot=report_snapshot,
                            )
                            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                "下载回测完整报告ZIP",
                                data=report_bytes,
                                file_name=f"wyckoff_backtest_report_{stamp}.zip", mime="application/zip",
                                key="download_wyckoff_backtest_report_zip", use_container_width=True,
                            )
                        st.caption("完整报告包含：交易明细、权益曲线、指标汇总、回测参数、设置快照。")


if __name__ == "__main__":
    main()
