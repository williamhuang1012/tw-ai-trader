"""Phase 1: Taiwan equity single-symbol swing trading research baseline.

Run: python main.py --stock 2330 --start 2016-01-01 --end 2026-10-14
Signals use end-of-day features; simulated fills happen at the NEXT session open.
This is research/backtesting, NOT live paper execution.
"""
import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FEATURES = ["ret1", "ret5", "ret10", "ma5_gap", "ma20_gap", "ma60_gap",
            "volatility20", "vol_ratio", "rsi14", "macd_gap", "macd_hist"]
HORIZON = 5
FEE = .001425
TAX = .003
SLIPPAGE = .0005
LOT = 1


def fetch_prices(stock: str, start: str, end: str) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    cache = DATA / f"{stock}_{start}_{end}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        params = {"dataset": "TaiwanStockPrice", "data_id": stock,
                  "start_date": start, "end_date": end}
        token = os.getenv("FINMIND_TOKEN", "").strip()
        if token:
            params["token"] = token
        response = requests.get("https://api.finmindtrade.com/api/v4/data",
                                params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != 200 or not payload.get("data"):
            raise RuntimeError(f"FinMind API error: {payload.get('msg', payload)}")
        df = pd.DataFrame(payload["data"])
        df.to_csv(cache, index=False)
    df = df.rename(columns={"max": "high", "min": "low",
                            "Trading_Volume": "volume"})
    required = ["date", "open", "high", "low", "close", "volume"]
    absent = set(required) - set(df.columns)
    if absent:
        raise ValueError(f"Missing price columns: {sorted(absent)}")
    df["date"] = pd.to_datetime(df["date"])
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    df = df.dropna(subset=required[1:])
    df = df.loc[(df["close"] > 0) & (df["open"] > 0)].copy()
    if len(df) < 550:
        raise ValueError("Need at least 550 daily price observations")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["close"]
    x["ret1"] = c.pct_change()
    x["ret5"] = c.pct_change(5)
    x["ret10"] = c.pct_change(10)
    for n in (5, 20, 60):
        x[f"ma{n}_gap"] = c / c.rolling(n).mean() - 1
    x["volatility20"] = x["ret1"].rolling(20).std()
    x["vol_ratio"] = x["volume"] / x["volume"].rolling(20).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    x["rsi14"] = 100 - 100 / (1 + rs)
    x.loc[(down == 0) & (up > 0), "rsi14"] = 100
    x.loc[(down == 0) & (up == 0), "rsi14"] = 50
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    x["macd_gap"] = macd / c
    x["macd_hist"] = (macd - signal) / c
    x["future_return"] = c.shift(-HORIZON) / c - 1
    x["target"] = (x["future_return"] > .02).astype(int)
    x = x.replace([np.inf, -np.inf], np.nan)
    # Preserve last HORIZON rows as unlabelled dates; never train on those.
    x = x.dropna(subset=FEATURES)
    return x


def backtest(test: pd.DataFrame, initial_cash: float, buy_prob: float,
             sell_prob: float, max_allocation: float):
    """Signal at day t close, fills at t+1 open; long-only, 1000-share lots."""
    cash = float(initial_cash)
    shares = 0
    entries = []
    curve = []
    dates = list(test.index)
    # Starting marked portfolio, before any order.
    curve.append({"date": dates[0], "equity": cash,
                  "close": float(test.iloc[0]["close"])})
    for i in range(1, len(test)):
        today = test.iloc[i]
        yesterday = test.iloc[i - 1]
        price_open = float(today["open"])
        price_close = float(today["close"])
        signal = float(yesterday["probability"])
        date = dates[i]
        if shares > 0 and signal < sell_prob:
            fill = price_open * (1 - SLIPPAGE)
            proceeds = shares * fill
            fees = max(20.0, proceeds * FEE)
            tax = proceeds * TAX
            cash += proceeds - fees - tax
            entries.append({"date": date, "side": "SELL", "shares": shares,
                            "fill": fill, "fees": fees, "tax": tax, "cash": cash,
                            "signal_probability": signal})
            shares = 0
        elif shares == 0 and signal >= buy_prob:
            fill = price_open * (1 + SLIPPAGE)
            budget = min(cash, initial_cash * max_allocation)
            num_lots = int(budget // (fill * LOT * (1 + FEE)))
            if num_lots > 0:
                qty = num_lots * LOT
                principal = qty * fill
                fees = max(20.0, principal * FEE)
                if principal + fees <= cash:
                    cash -= principal + fees
                    shares = qty
                    entries.append({"date": date, "side": "BUY", "shares": shares,
                                    "fill": fill, "fees": fees, "tax": 0.0,
                                    "cash": cash, "signal_probability": signal})
        curve.append({"date": date, "equity": cash + shares * price_close,
                      "close": price_close})
    equity = pd.DataFrame(curve).set_index("date")
    trades = pd.DataFrame(entries)
    daily = equity["equity"].pct_change().fillna(0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, .01)
    total_ret = equity["equity"].iloc[-1] / initial_cash - 1
    benchmark = equity["close"].iloc[-1] / equity["close"].iloc[0] - 1
    dd = equity["equity"] / equity["equity"].cummax() - 1
    std = daily.std()
    metrics = {
        "total_return": float(total_ret),
        "buy_hold_price_return_ex_dividend": float(benchmark),
        "cagr": float((1 + total_ret) ** (1 / years) - 1),
        "max_drawdown": float(dd.min()),
        "sharpe_annualized_rf0": float(np.sqrt(252) * daily.mean() / std) if std > 0 else np.nan,
        "orders": int(len(trades)),
        "ending_shares": int(shares),
        "ending_cash": float(cash),
    }
    return equity, trades, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default="2330")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-10-14")
    ap.add_argument("--cash", type=float, default=1_000_000)
    ap.add_argument("--buy-prob", type=float, default=.60)
    ap.add_argument("--sell-prob", type=float, default=.45)
    ap.add_argument("--max-allocation", type=float, default=.30)
    args = ap.parse_args()
    if not (0 <= args.sell_prob < args.buy_prob <= 1):
        ap.error("Require 0 <= sell-prob < buy-prob <= 1")
    if args.cash <= 0 or not (0 < args.max_allocation <= 1):
        ap.error("Cash must be > 0 and max-allocation in (0,1]")
    load_dotenv(ROOT / ".env")
    OUT.mkdir(exist_ok=True)
    df = engineer(fetch_prices(args.stock, args.start, args.end))
    n = len(df)
    train_cut, val_cut = int(n * .70), int(n * .85)
    # Train rows whose future-close target is fully inside the train interval.
    train = df.iloc[:train_cut - HORIZON].dropna(subset=["future_return"])
    val = df.iloc[train_cut:val_cut - HORIZON].dropna(subset=["future_return"])
    # Test permits last unlabelled dates for execution and PnL calculation.
    test = df.iloc[val_cut:].copy()
    if len(train) < 200 or len(val) < 30 or len(test) < 30:
        raise ValueError("Not enough samples in date splits")
    if train["target"].nunique() < 2:
        raise ValueError("Training target has only one class")
    model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=.04,
                          subsample=.8, colsample_bytree=.9,
                          objective="binary:logistic", eval_metric="logloss",
                          random_state=42, n_jobs=4)
    model.fit(train[FEATURES], train["target"])
    for name, part in (("validation", val), ("test", test.dropna(subset=["future_return"]))):
        p = model.predict_proba(part[FEATURES])[:, 1]
        a = accuracy_score(part["target"], p >= .5)
        auc = roc_auc_score(part["target"], p) if part["target"].nunique() == 2 else np.nan
        print(f"{name}: n={len(part)} accuracy={a:.3f}, AUC={auc:.3f}, positive_rate={part['target'].mean():.3f}")
    test["probability"] = model.predict_proba(test[FEATURES])[:, 1]
    equity, orders, metrics = backtest(test, args.cash, args.buy_prob,
                                       args.sell_prob, args.max_allocation)
    prefix = OUT / args.stock
    test[["close", "probability", "target", "future_return"]].to_csv(f"{prefix}_predictions.csv", encoding="utf-8-sig")
    equity.to_csv(f"{prefix}_equity.csv", encoding="utf-8-sig")
    orders.to_csv(f"{prefix}_orders.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"model": model, "features": FEATURES}, f"{prefix}_model.joblib")
    pd.Series(metrics).to_csv(f"{prefix}_metrics.csv", header=["value"])
    ax = (equity["equity"] / args.cash).plot(figsize=(10, 5), label="AI strategy (net costs)")
    (equity["close"] / equity["close"].iloc[0]).plot(ax=ax, label="Buy & hold (price only)")
    ax.set_title(f"{args.stock} - Out-of-sample simulation")
    ax.set_ylabel("Normalized wealth / price")
    ax.grid(alpha=.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_equity.png", dpi=150)
    plt.close()
    print("Test window:", str(test.index[0].date()), "to", str(test.index[-1].date()))
    print(pd.Series(metrics).to_string())
    print("Outputs written to:", OUT)
    print("NOTE: backtest only; raw prices not adjusted for corporate actions/dividends; no halt/limit fill model.")


if __name__ == "__main__":
    main()
