#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def customize_splines(ax: plt.axis) -> plt.axis:
    ax.spines["left"].set_edgecolor("black"),
    ax.spines["left"].set_linewidth(2),
    ax.spines["left"].set_zorder(3),
    ax.spines["bottom"].set_edgecolor("black"),
    ax.spines["bottom"].set_linewidth(2),
    ax.spines["bottom"].set_zorder(3),
    ax.spines["top"].set_edgecolor("lightgrey"),
    ax.spines["top"].set_linewidth(1),
    ax.spines["right"].set_edgecolor("lightgrey"),
    ax.spines["right"].set_linewidth(1),
    return ax


def plot_store_item_grid(
    df, store_item_dict, year_start, year_end, ts_name, fig_size=(12, 40)
):
    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(10, 1, hspace=0, wspace=0)
    ax0 = fig.add_subplot(grid[0, 0])
    stores = store_item_dict["store"]
    items = store_item_dict["item"]
    for k in range(10):
        if k > 0:
            ax = fig.add_subplot(grid[k, 0], xticklabels=[], sharex=ax0)
        else:
            ax = ax0
        ts = df[
            (df["store"] == stores[k]) & (df["item"] == items[k])
        ].set_index("date")
        ts = ts.loc[year_start:year_end][ts_name]
        ts.plot(ax=ax, color="blue")
        ax.grid(color="lightgrey")
        ax.set_title(
            f"Store {stores[k]}, item {items[k]} in {year_start}-{year_end}",
            loc="left",
            fontweight="bold",
            x=0.01,
            y=0.85,
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        _ = customize_splines(ax)


def show_splits(
    train_cv,
    test_cv,
    fold_number,
    ts_name,
    n_splits,
    ptitle,
    ax,
    lw,
):
    train_cv_slice = train_cv.loc[slice(ts_name), slice(None)]
    test_cv_slice = test_cv.loc[slice(ts_name), slice(None)]
    ax.plot(
        train_cv_slice.index.get_level_values(1),
        pd.Series([fold_number] * len(train_cv_slice), dtype=int),
        color="black",
        alpha=1.0,
        label="train",
        lw=lw,
    )
    ax.plot(
        test_cv_slice.index.get_level_values(1),
        pd.Series([fold_number] * len(test_cv_slice), dtype=int),
        color="orange",
        alpha=1.0,
        label="test",
        lw=lw,
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_to_annotate = 1 if "Expanding" in ptitle else n_splits
    if fold_number == ax_to_annotate:
        ax.set_ylabel("Fold Number", fontsize=16)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.00, 1.05),
            ncol=1,
            handletextpad=0.4,
            columnspacing=0.6,
            frameon=False,
            prop={"size": 12},
        )
        ax.grid()
        ax.set_title(
            f"CV folds using {ptitle} with {n_splits} folds",
            fontweight="bold",
            fontsize=14,
            loc="left",
        )
        customize_splines(ax)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
