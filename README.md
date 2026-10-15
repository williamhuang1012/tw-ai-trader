# 台股 XGBoost 波段交易：第一階段

此專案是**歷史回測研究原型**，不是連接券商的即時模擬帳戶。每日收盤產生訊號，下一個交易日開盤模擬成交。

## 安裝

Python 3.10+ (建議 3.11)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

可選：複製 `.env.example` 為 `.env`，填入 `FINMIND_TOKEN=...`。無 token 也可嘗試有限額的 API。

## 執行

```bash
python main.py --stock 2330 --start 2016-01-01 --end 2026-10-14
```

輸出 `outputs/2330_predictions.csv`、`2330_orders.csv`、`2330_equity.csv`、`2330_metrics.csv`、`2330_equity.png`、`2330_model.joblib`。

## 設計與限制

- 資料源 FinMind `TaiwanStockPrice`，使用原始價位進行模擬成交。
- 特徵：報酬率、MA 價差、波動度、量比、RSI、MACD。
- 標籤：未來 5 個交易日收盤報酬 > 2%；模型預測不等同實際成交後 5 日報酬。
- 日期序列 70/15/15 切分；訓練和驗證尾端去除 5 日以隔離前瞻標籤。門檻是固定的假設，非最佳化結果。
- 單檔股票、現股整張(1000 股)、不融資、不放空；每次最多以起始資金 30% 進場，空手再買。
- 手續費每側 0.1425%、最低 20 元的近似假設，賣出稅 0.3%、每側滑價 0.05%。
- 使用 raw (未還原) OHLCV，不含配息、除權息調整，因此長期績效會失真；尚未建模停牌、漲跌停、委託未成交、股價跳空極端情況、整張例外、成交量衝擊。**不要以本版績效直接決定實盤投資**。
- `orders.csv` 為訂單執行紀錄，報酬曲線終值按最後收盤價計算持股市值；沒有強制最後清倉。
- 資料最後幾天無未來標籤，仍可作回測訊號，但不能納入評分。
- 下一步：還原價格+公司行動/現金股利、純前推 Walk-forward、更多股票/交易日/存活者偏差、模擬撮合、paper account 狀態持久化。
