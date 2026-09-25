"""PNG-графики отчёта (matplotlib). Один y-масштаб на график, тонкие линии, приглушённые оси,
категориальные цвета в фиксированном порядке (эталонная палитра), дивергентная шкала для тепловой карты."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#c0392b", "#f2c4b8", "#efefec", "#b9d3f0", "#1f5fae"])


def _style(ax, title: str, ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT2, fontsize=8)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def line_chart(series: dict[str, pd.Series], path, title: str, ylabel: str = "", pct: bool = False):
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (name, s) in enumerate(list(series.items())[:8]):
        ax.plot(s.index, s.values * (100 if pct else 1), color=SERIES[k], linewidth=1.6, label=name)
    _style(ax, title, ylabel)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2)
    _save(fig, path)


def drawdown_chart(dd: dict[str, pd.Series], path, title: str):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for k, (name, s) in enumerate(list(dd.items())[:8]):
        ax.fill_between(s.index, s.values * 100, 0, color=SERIES[k], alpha=0.25 if len(dd) > 1 else 0.35,
                        linewidth=0)
        ax.plot(s.index, s.values * 100, color=SERIES[k], linewidth=1.2, label=name)
    _style(ax, title, "просадка, %")
    if len(dd) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2)
    _save(fig, path)


def monthly_heatmap(table: pd.DataFrame, path, title: str):
    t = table.reindex(columns=range(1, 13)) * 100
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(t) + 1.4))
    vals = t.to_numpy(float)
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim - 1e-9, lim + 1e-9), aspect="auto")
    ax.set_xticks(range(12), ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"])
    ax.set_yticks(range(len(t)), [str(y) for y in t.index])
    for i in range(vals.shape[0]):
        for j in range(12):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=7, color=TEXT)
    _style(ax, title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).ax.tick_params(labelsize=7, colors=TEXT2)
    _save(fig, path)


def histogram(values: np.ndarray, path, title: str, xlabel: str, marks: dict[str, float] | None = None):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(values, bins=60, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=0.5)
    _style(ax, title)
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=8)
    for k, (name, x) in enumerate((marks or {}).items()):
        ax.axvline(x, color=TEXT2, linewidth=1, linestyle="--")
        ax.text(x, ax.get_ylim()[1] * (0.92 - 0.1 * k), f" {name}", color=TEXT2, fontsize=8)
    _save(fig, path)


def heat_surface(df: pd.DataFrame, path, title: str, xlabel: str, ylabel: str, fmt: str = "{:.2f}"):
    vals = df.to_numpy(float)
    fig, ax = plt.subplots(figsize=(1.0 * df.shape[1] + 3, 0.5 * df.shape[0] + 1.8))
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim - 1e-9, lim + 1e-9), aspect="auto")
    ax.set_xticks(range(df.shape[1]), [str(c) for c in df.columns])
    ax.set_yticks(range(df.shape[0]), [str(r) for r in df.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=7, color=TEXT)
    _style(ax, title)
    ax.grid(False)
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).ax.tick_params(labelsize=7, colors=TEXT2)
    _save(fig, path)


def bar_chart(labels: list[str], values: list[float], path, title: str, xlabel: str,
              errors: list[float] | None = None):
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(labels) + 1.4))
    y = np.arange(len(labels))
    colors = [SERIES[0] if v >= 0 else SERIES[1] for v in values]
    ax.barh(y, values, color=colors, height=0.6, xerr=errors, error_kw=dict(ecolor=TEXT2, lw=0.8, capsize=2))
    ax.axvline(0, color=TEXT2, linewidth=0.8)
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    _style(ax, title)
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=8)
    _save(fig, path)
