"""Stage 1: распределение результатов скрининга (без выбора!) -> research/stage1/*.csv, *.md.

Все метрики — на периоде разработки 2025-09-25 … 2026-06-30, издержки NORMAL (Трейдер), доходность
в долях капитала при номинале позиции = 1× капитала (без плеча и сложного процента).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import data as D  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "s1"
OUT = D.ROOT / "research/stage1"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    files = sorted((D.CACHE / "screen" / TAG).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["active"] = df["trades"] >= 10
    n = len(df)
    mcols = [c for c in df.columns if c.startswith("m20")]
    # полная таблица конфигураций (компактно) — хранится в git как журнал Stage 1
    keep = ["code", "tf", "family", "group", "params", "exit", "eod", "direction", "trades", "win", "pf", "avg_trade",
            "hold_bars", "total", "sharpe", "sortino", "mdd", "pmr", "m_min", "m_median", "m_mean", "m_std", "score"]
    slim = df[keep + mcols].copy()
    for c in slim.select_dtypes("float64").columns:
        slim[c] = slim[c].astype("float32")
    slim.to_parquet(OUT / f"{TAG}_all_configs.parquet", index=False, compression="zstd")

    def agg(g):
        a = g[g["active"]]
        return pd.Series(dict(configs=len(g), active=len(a),
                              share_positive=float((a["total"] > 0).mean()) if len(a) else np.nan,
                              share_sharpe_gt1=float((a["sharpe"] > 1).mean()) if len(a) else np.nan,
                              median_sharpe=float(a["sharpe"].median()) if len(a) else np.nan,
                              p90_sharpe=float(a["sharpe"].quantile(0.9)) if len(a) else np.nan,
                              best_sharpe=float(a["sharpe"].max()) if len(a) else np.nan,
                              median_total=float(a["total"].median()) if len(a) else np.nan,
                              share_pmr_ge_08=float((a["pmr"] >= 0.8).mean()) if len(a) else np.nan,
                              median_avg_trade_bp=float(1e4 * a["avg_trade"].median()) if len(a) else np.nan))

    by_fam = df.groupby(["group", "family"]).apply(agg).reset_index().sort_values("median_sharpe", ascending=False)
    by_tf = df.groupby("tf").apply(agg).reset_index()
    by_code = df.groupby("code").apply(agg).reset_index().sort_values("median_sharpe", ascending=False)
    by_dir = df.groupby(["direction"]).apply(agg).reset_index()
    by_exit = df.groupby(["exit"]).apply(agg).reset_index()
    by_eod = df.groupby(["eod"]).apply(agg).reset_index()
    by_fam_tf = df.groupby(["family", "tf"]).apply(agg).reset_index()
    for name, t in (("by_family", by_fam), ("by_tf", by_tf), ("by_code", by_code), ("by_direction", by_dir),
                    ("by_exit", by_exit), ("by_eod", by_eod), ("by_family_tf", by_fam_tf)):
        t.to_csv(OUT / f"{TAG}_{name}.csv", index=False)
    # гистограмма Sharpe
    a = df[df["active"]]
    hist = np.histogram(a["sharpe"].clip(-10, 10), bins=np.arange(-10, 10.5, 0.5))
    pd.DataFrame({"sharpe_from": hist[1][:-1], "count": hist[0]}).to_csv(OUT / f"{TAG}_sharpe_hist.csv", index=False)
    top = a.sort_values("sharpe", ascending=False).head(300)
    top[keep + mcols].to_csv(OUT / f"{TAG}_top300_by_sharpe_INSAMPLE.csv", index=False)
    lines = [f"# Stage 1 ({TAG}): распределение результатов, период разработки (in-sample)", "",
             f"Конфигураций: **{n:,}**; с ≥10 сделками: **{int(df['active'].sum()):,}**; семейств: {df['family'].nunique()}; "
             f"инструментов: {df['code'].nunique()}; таймфреймов: {df['tf'].nunique()}.", "",
             f"Доля конфигураций (≥10 сделок) с положительным итогом: **{(a['total'] > 0).mean():.1%}**; "
             f"с Sharpe > 1: {(a['sharpe'] > 1).mean():.1%}; > 2: {(a['sharpe'] > 2).mean():.1%}; > 3: {(a['sharpe'] > 3).mean():.2%}.",
             "", "Это in-sample: лучшие строки здесь — результат перебора, а не доказательство преимущества. "
             "Честная оценка — walk-forward (Stage 3).", "", "## По таймфреймам", "", by_tf.round(3).to_markdown(index=False),
             "", "## По семействам", "", by_fam.round(3).to_markdown(index=False), "", "## По инструментам", "",
             by_code.round(3).to_markdown(index=False), "", "## Направление / выходы / перенос", "",
             by_dir.round(3).to_markdown(index=False), "", by_exit.round(3).to_markdown(index=False), "",
             by_eod.round(3).to_markdown(index=False)]
    (OUT / f"{TAG}_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
