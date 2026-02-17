"""Reusable plotting functions for the stock prediction project."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

from .config import MODEL_NAMES, MODEL_COLORS, MODEL_MARKERS


def setup_style():
    """Apply the project's default Matplotlib / Seaborn style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_normalized_prices(stock_data, representative_stocks):
    """Normalized price evolution (base = 100) for representative stocks.

    Parameters
    ----------
    stock_data : dict[str, DataFrame]
    representative_stocks : dict[str, str]  — {sector: ticker}
    """
    n = len(representative_stocks)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (sector, ticker) in enumerate(representative_stocks.items()):
        if ticker not in stock_data:
            continue
        df = stock_data[ticker]
        norm = (df["Close"] / df["Close"].iloc[0]) * 100
        axes[idx].plot(df.index, norm, linewidth=1.5, color="steelblue")
        axes[idx].axhline(y=100, color="red", linestyle="--", alpha=0.5)
        axes[idx].set_title(f"{sector}: {ticker}", fontweight="bold")
        axes[idx].set_ylabel("Normalized Price")
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Normalized Price Evolution by Sector", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.show()


def plot_return_distributions(stock_data, representative_stocks):
    """Histogram of daily returns with stats box."""
    n = len(representative_stocks)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (sector, ticker) in enumerate(representative_stocks.items()):
        if ticker not in stock_data:
            continue
        rets = stock_data[ticker]["Close"].pct_change().dropna() * 100
        ax = axes[idx]
        ax.hist(rets, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(rets.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {rets.mean():.3f}%")
        ax.set_title(f"{ticker} ({sector})", fontweight="bold")
        ax.set_xlabel("Daily Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        stats_text = f"Std: {rets.std():.3f}%\nSkew: {rets.skew():.3f}\nKurt: {rets.kurtosis():.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Daily Return Distributions by Sector", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr_matrix, title="Correlation Matrix"):
    """Generic heatmap for a correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Indicator plots
# ---------------------------------------------------------------------------

def plot_indicator_with_price(dates, prices, indicator, indicator_name,
                              overbought=None, oversold=None, ticker=""):
    """Two-panel chart: price on top, indicator on bottom."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    ax1.plot(dates, prices, linewidth=1.5, color="black", label="Close Price")
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{ticker} Price and {indicator_name}", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(dates, indicator, linewidth=1.5, color="blue", label=indicator_name)
    if overbought is not None:
        ax2.axhline(y=overbought, color="red", linestyle="--", alpha=0.7, label=f"Overbought ({overbought})")
        ax2.fill_between(dates, overbought, indicator.max() * 1.05 if hasattr(indicator, "max") else 100,
                         alpha=0.15, color="red")
    if oversold is not None:
        ax2.axhline(y=oversold, color="green", linestyle="--", alpha=0.7, label=f"Oversold ({oversold})")
        ax2.fill_between(dates, indicator.min() * 0.95 if hasattr(indicator, "min") else 0, oversold,
                         alpha=0.15, color="green")

    ax2.set_ylabel(indicator_name)
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Model comparison plots
# ---------------------------------------------------------------------------

def plot_kfold_comparison(comparison_df, windows, analysis_ticker):
    """Six-panel chart comparing Standard vs Purged K-Fold across metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F-Score", "AUC"]
    method_colors = {"Standard K-Fold": "red", "Purged K-Fold": "teal"}

    x = np.arange(len(windows))
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for i, mn in enumerate(MODEL_NAMES):
            offset = (i - (len(MODEL_NAMES) - 1) / 2) * 0.08
            for method, color in method_colors.items():
                vals = comparison_df[
                    (comparison_df["Model"] == mn) & (comparison_df["Method"] == method)
                ][metric].values
                ax.plot(x + offset, vals, marker=MODEL_MARKERS[mn], linestyle="None",
                        markersize=6, color=color, alpha=0.4)

        for method, color in method_colors.items():
            avg = comparison_df[comparison_df["Method"] == method].groupby("Window")[metric].mean()
            marker = "o" if method == "Standard K-Fold" else "s"
            ax.plot(x, avg.values, f"{marker}-", color=color, linewidth=2, markersize=8,
                    label=f"{method} (avg)")

        ax.set_xlabel("Trading Window (days)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(windows)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Validation Method Comparison: {analysis_ticker}\nStandard vs Purged K-Fold",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, feature_importances, ticker, window):
    """Bar charts of feature importance for each model."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Feature Importance: {ticker} | Window: {window} days",
                 fontsize=14, fontweight="bold")

    for idx, model_name in enumerate(MODEL_NAMES):
        if idx >= 5:
            break
        ax = axes[idx // 3, idx % 3]
        imp_pct = feature_importances[model_name]["percentage"]
        order = np.argsort(imp_pct)[::-1]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_names)))
        ax.barh([feature_names[i] for i in order], imp_pct[order], color=colors)
        ax.set_xlabel("Importance (%)")
        ax.set_title(model_name, fontweight="bold")
        for j, v in enumerate(imp_pct[order]):
            ax.text(v + 0.5, j, f"{v:.1f}%", va="center", fontsize=9)

    # Hide last subplot if unused
    if len(MODEL_NAMES) < 6:
        fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.show()


def plot_roc_curves(trained_models):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, mdata in trained_models.items():
        y_test = mdata["y_test"]
        y_proba = mdata["y_proba"]
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        from sklearn.metrics import roc_auc_score
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, linewidth=2, color=MODEL_COLORS[model_name],
                label=f"{model_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(trained_models, ticker, window):
    """Confusion matrices for each model."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(f"Confusion Matrices: {ticker} | Window: {window} days",
                 fontsize=14, fontweight="bold")

    for idx, model_name in enumerate(MODEL_NAMES):
        if idx >= 5:
            break
        ax = axes[idx // 3, idx % 3]
        mdata = trained_models[model_name]
        cm = confusion_matrix(mdata["y_test"], mdata["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    if len(MODEL_NAMES) < 6:
        fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_window(single_stock_df, ticker):
    """Line chart of accuracy across trading windows for each model."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for model in MODEL_NAMES:
        md = single_stock_df[single_stock_df["Model"] == model]
        ax.plot(md["Window"], md["Accuracy"] * 100, marker=MODEL_MARKERS[model],
                linewidth=2, markersize=8, color=MODEL_COLORS[model], label=model)

    ax.set_xlabel("Trading Window (days)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy vs Trading Window — {ticker}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sector_bar_chart(sector_df, sectors):
    """Grouped bar chart of average accuracy by sector and model."""
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(sectors))
    width = 0.15

    for i, model in enumerate(MODEL_NAMES):
        accs = []
        for sector in sectors:
            row = sector_df[(sector_df["Sector"] == sector) & (sector_df["Model"] == model)]
            accs.append(row["Avg_Accuracy"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, accs, width, label=model, color=MODEL_COLORS[model])

    ax.set_xlabel("Sector", fontsize=12)
    ax.set_ylabel("Average Accuracy (%)", fontsize=12)
    ax.set_title("Model Performance by Sector", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(sectors, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_sector_heatmap(sector_df, sectors):
    """Heatmap of accuracy: sectors x models."""
    matrix = []
    for sector in sectors:
        row = []
        for model in MODEL_NAMES:
            sub = sector_df[(sector_df["Sector"] == sector) & (sector_df["Model"] == model)]
            row.append(sub["Avg_Accuracy"].values[0] if len(sub) > 0 else 0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=MODEL_NAMES, yticklabels=sectors, ax=ax)
    ax.set_title("Average Accuracy (%) by Sector and Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_window_effect(window_df, windows, results_df=None):
    """Line plot with confidence bands + box plot for window effect."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Line plot
    ax1 = axes[0]
    for model in MODEL_NAMES:
        md = window_df[window_df["Model"] == model].sort_values("Window")
        ax1.plot(md["Window"], md["Avg_Accuracy"], marker=MODEL_MARKERS[model],
                 linewidth=2, markersize=8, color=MODEL_COLORS[model], label=model)
        ax1.fill_between(md["Window"],
                         md["Avg_Accuracy"] - md["Std"],
                         md["Avg_Accuracy"] + md["Std"],
                         alpha=0.15, color=MODEL_COLORS[model])
    ax1.set_xlabel("Trading Window (days)")
    ax1.set_ylabel("Average Accuracy (%)")
    ax1.set_title("Accuracy vs Window (with ±1σ)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Box plot — per-stock accuracy distribution by window
    ax2 = axes[1]
    if results_df is not None and len(results_df) > 0:
        box_data = []
        box_labels = []
        for w in sorted(windows):
            wd = results_df[results_df["Window"] == w]
            if len(wd) > 0:
                box_data.append(wd["Accuracy"].values * 100)
                box_labels.append(f"{w}d")
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            colors_bp = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(box_data)))
            for patch, c in zip(bp["boxes"], colors_bp):
                patch.set_facecolor(c)
    ax2.set_title("Accuracy Distribution by Window")
    ax2.set_xlabel("Trading Window (days)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_window_heatmap(window_df, windows):
    """Heatmap of accuracy: windows x models."""
    matrix = []
    for window in windows:
        row = []
        for model in MODEL_NAMES:
            sub = window_df[(window_df["Window"] == window) & (window_df["Model"] == model)]
            row.append(sub["Avg_Accuracy"].values[0] if len(sub) > 0 else 0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=MODEL_NAMES, yticklabels=[f"{w}d" for w in windows], ax=ax)
    ax.set_title("Average Accuracy (%) by Window and Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_portfolio_evolution(sector_portfolios):
    """Line chart of portfolio cumulative value per sector."""
    plt.figure(figsize=(15, 8))
    for sector, pdf in sector_portfolios.items():
        plt.plot(pdf.index, pdf["Close"], linewidth=2, label=sector, alpha=0.8)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Sector Portfolio Evolution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_individual_vs_portfolio(comparison_df, sectors):
    """Side-by-side accuracy comparison for individual stocks vs portfolios."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle("Average Accuracy: Individual Stocks vs Portfolios\nAcross Trading Windows",
                 fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODEL_NAMES):
        if idx >= len(axes):
            break
        ax = axes[idx]
        md = comparison_df[comparison_df["Model"] == model]
        for t in md["Type"].unique():
            sub = md[md["Type"] == t].groupby("Window")["Accuracy"].mean()
            marker = "o" if "Individual" in t else "s"
            ax.plot(sub.index, sub.values * 100, marker=marker, linewidth=2, label=t)
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Window (days)")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if len(MODEL_NAMES) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()
