#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import altair as alt
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


def altair_datetime_heatmap_grid(
    data,
    date_col,
    yvar,
    ptitle,
    marker_linewidth,
    agg="mean",
    ptitle_x_loc=15,
    chart_separation=5,
    fig_width=240,
    fig_half_heights=(350, 370),
    fig_height=750,
    cmap="yelloworangered",
    scale="linear",
):
    hm_month_date = (
        alt.Chart(
            data,
            title=ptitle,
        )
        .mark_rect(stroke="white", strokeWidth=marker_linewidth)
        .encode(
            y=alt.Y(
                f"date({date_col}):N",
                title=None,
                axis=alt.Axis(ticks=False),
            ),
            x=alt.X(
                f"month({date_col}):N",
                title=None,
                axis=alt.Axis(ticks=False),
            ),
            color=alt.Color(
                f"{agg}({yvar}):Q",
                scale=alt.Scale(type=scale, scheme=cmap),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"month({date_col}):T", title="Month"),
                alt.Tooltip(f"date({date_col}):T", title="Day of Month"),
                alt.Tooltip(
                    f"{agg}({yvar}):Q", title=f"{agg.title()} {yvar.title()}"
                ),
            ],
        )
        .properties(height=fig_height, width=fig_width)
    )
    hm_mon_dow = (
        alt.Chart(data)
        .mark_rect(stroke="white", strokeWidth=marker_linewidth)
        .encode(
            y=alt.Y(
                f"day({date_col}):N",
                title=None,
                axis=alt.Axis(orient="right", ticks=False),
            ),
            x=alt.X(
                f"month({date_col}):N",
                title=None,
                axis=alt.Axis(ticks=False),
            ),
            color=alt.Color(
                f"{agg}({yvar}):Q",
                scale=alt.Scale(type=scale, scheme=cmap),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"month({date_col}):T", title="Month"),
                alt.Tooltip(f"day({date_col}):T", title="Weekday"),
                alt.Tooltip(
                    f"{agg}({yvar}):Q", title=f"{agg.title()} {yvar.title()}"
                ),
            ],
        )
        .properties(height=fig_half_heights[0], width=fig_width)
    )
    hm_q_yr = (
        alt.Chart(data)
        .mark_rect(stroke="white", strokeWidth=marker_linewidth)
        .encode(
            y=alt.Y(
                f"year({date_col}):N",
                title=None,
                axis=alt.Axis(orient="right", ticks=False),
            ),
            x=alt.X(
                f"quarter({date_col}):N",
                title=None,
                axis=alt.Axis(ticks=False, labelAngle=0),
            ),
            color=alt.Color(
                f"{agg}({yvar}):Q",
                scale=alt.Scale(type=scale, scheme=cmap),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"year({date_col}):T", title="Year"),
                alt.Tooltip(f"quarter({date_col}):T", title="Quarter"),
                alt.Tooltip(
                    f"{agg}({yvar}):Q", title=f"{agg.title()} {yvar.title()}"
                ),
            ],
        )
        .properties(height=fig_half_heights[1], width=fig_width)
    )
    hm_yr_wk = (
        alt.Chart(data)
        .mark_rect(stroke="white", strokeWidth=marker_linewidth)
        .encode(
            y=alt.Y(
                "week:N",
                title=None,
                axis=alt.Axis(orient="right", ticks=False),
            ),
            x=alt.X(
                f"year({date_col}):N",
                title=None,
                axis=alt.Axis(ticks=False, labelAngle=0),
            ),
            color=alt.Color(
                f"{agg}({yvar}):Q",
                scale=alt.Scale(type=scale, scheme=cmap),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"year({date_col}):T", title="Year"),
                alt.Tooltip("week:O", title="Week"),
                alt.Tooltip(
                    f"{agg}({yvar}):Q", title=f"{agg.title()} {yvar.title()}"
                ),
            ],
        )
        .properties(height=fig_height, width=fig_width)
    )
    heatmap_grid = (
        alt.hconcat(hm_month_date, alt.vconcat(hm_mon_dow, hm_q_yr), hm_yr_wk)
        .configure_view(strokeWidth=0)
        .configure_axis(domain=False, labelFontSize=12, titleFontSize=14)
        .configure_concat(spacing=chart_separation)
        .configure_title(anchor="start", dx=ptitle_x_loc, fontSize=14)
    )
    return heatmap_grid


def plot_auto_correlation_plot(
    df: pd.DataFrame,
    yvar: str,
    tooltip: list,
    marker_size: int = 80,
    vertical_line_width: float = 1.5,
    marker_linewidth: int = 1,
    ci_band_opacity: float = 0.5,
    zero_y_linewidth: float = 0.5,
    fig_size: tuple = (300, 300),
):
    df["zero_y"] = pd.Series([0] * len(df))
    bars = (
        alt.Chart(df)
        .mark_rule(color="black", strokeWidth=vertical_line_width)
        .encode(
            x=alt.X("lag:Q"),
            y=alt.Y(f"{yvar}:Q", title=""),
        )
    )
    circles = (
        alt.Chart(df)
        .mark_circle(
            color="blue",
            size=marker_size,
            strokeWidth=marker_linewidth,
            stroke="black",
        )
        .encode(
            x=alt.X("lag:Q"),
            y=alt.Y(f"{yvar}:Q", title=""),
            tooltip=tooltip,
        )
    )
    ci = (
        alt.Chart(df, title=yvar.upper())
        .mark_area(opacity=ci_band_opacity)
        .encode(
            x=alt.X(
                "lag:Q",
                axis=alt.Axis(
                    title="lag",
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            y=alt.Y(
                "y_min:Q",
                axis=alt.Axis(
                    title="",
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            y2="y_max:Q",
        )
    )
    zero_rule = (
        alt.Chart(df)
        .mark_rule(size=zero_y_linewidth, color="blue", opacity=1)
        .encode(y="zero_y")
    )
    combined_chart = alt.layer(ci, zero_rule, bars, circles).properties(
        width=fig_size[0], height=fig_size[1]
    )
    return combined_chart


def plot_acf_pacf(
    df: pd.DataFrame,
    yvar: str,
    n_lags: int,
    corr_plots_tooltip: dict,
    alpha: float = 0.05,
    ci_band_opacity: float = 0.5,
    corr_plots_wanted: list = ["acf", "pacf"],
    marker_size=80,
    vertical_line_width: int = 2,
    marker_linewidth: int = 1,
    zero_y_linewidth: int = 0.5,
    fig_size: tuple = (325, 275),
) -> alt.Chart:
    corrs = ["acf"]
    a_cf, acf_ci = acf(df[yvar], nlags=n_lags - 1, fft=False, alpha=alpha)
    corr_func_values = [a_cf]
    corr_func_ci_values = [acf_ci]
    if "pacf" in corr_plots_wanted:
        p_acf, pacf_ci = pacf(df[yvar], nlags=n_lags - 1, alpha=alpha)
        corrs += ["pacf"]
        corr_func_values += [p_acf]
        corr_func_ci_values += [pacf_ci]
    charts = {}
    for corr_func_name, corr_func_calc, corr_func_ci in zip(
        corrs, corr_func_values, corr_func_ci_values
    ):
        df_cf = pd.DataFrame(corr_func_ci, columns=["low_ci", "high_ci"])
        df_cf["lag_min"] = 1
        df_cf["lag_max"] = n_lags
        df_cf["lag"] = range(1, n_lags + 1)
        df_cf[corr_func_name] = corr_func_calc
        df_cf["high_ci - low_ci"] = df_cf["high_ci"] - df_cf["low_ci"]
        df_cf[f"blue_shading = {corr_func_name} - low_ci"] = (
            df_cf[corr_func_name] - df_cf["low_ci"]
        )
        df_cf["blue_shading = (high_ci - low_ci) / 2"] = (
            df_cf["high_ci - low_ci"] / 2
        )
        df_cf["y_min"] = 0 - df_cf["high_ci - low_ci"] / 2
        df_cf["y_max"] = 0 + df_cf["high_ci - low_ci"] / 2
        charts[corr_func_name] = plot_auto_correlation_plot(
            df_cf,
            corr_func_name,
            corr_plots_tooltip[corr_func_name],
            marker_size,
            vertical_line_width,
            marker_linewidth,
            ci_band_opacity,
            zero_y_linewidth,
            fig_size=fig_size,
        )
    if "pacf" in corr_plots_wanted:
        combined_chart = alt.hconcat(charts["acf"], charts["pacf"])
        return combined_chart
    else:
        return charts["acf"]


def plot_ts(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    tooltip: list = [
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("value:Q", title="Value", format=".2f"),
    ],
    roll_window: int = 5,
    ts_linewidth: int = 1.5,
    roll_stat_linewidth: int = 3,
    roll_stat_colors: list = ["red", "black"],
    fig_size: tuple = (350, 250),
) -> alt.Chart:
    ts = (
        alt.Chart(df, title=f"{yvar.title()} Time Series")
        .mark_line(color="blue", strokeWidth=ts_linewidth)
        .encode(
            x=alt.X(
                f"{xvar}:T",
                axis=alt.Axis(
                    title="",
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            y=alt.Y(
                f"{yvar}:Q",
                axis=alt.Axis(
                    title="",
                    labels=True,
                    domainWidth=2,
                    domainColor="black",
                    domainOpacity=1,
                ),
            ),
            tooltip=tooltip,
        )
    )
    roll_stats_plots = {}
    for stat_type, roll_stat_color, roll_stat_thickness in zip(
        ["mean", "stdev"],
        roll_stat_colors,
        [roll_stat_linewidth, roll_stat_linewidth - 1],
    ):
        roll_stat_tooltip = [
            alt.Tooltip(
                "rolling_statistic:Q",
                title=f"{roll_window}-obs roll. {stat_type}",
                format=".1f",
            )
        ]
        roll_stat_line = (
            alt.Chart(df)
            .mark_line(color=roll_stat_color, strokeWidth=roll_stat_thickness)
            .transform_window(
                rolling_statistic=f"{stat_type}({yvar})",
                frame=[-int(roll_window / 2), int(roll_window / 2)],
            )
            .encode(
                x=alt.X(
                    f"{xvar}:T",
                    axis=alt.Axis(
                        title="",
                        labels=True,
                        domainWidth=2,
                        domainColor="black",
                        domainOpacity=1,
                    ),
                ),
                y=alt.Y(
                    "rolling_statistic:Q",
                    axis=alt.Axis(
                        title="",
                        labels=True,
                        domainWidth=2,
                        domainColor="black",
                        domainOpacity=1,
                    ),
                ),
                tooltip=tooltip + roll_stat_tooltip,
            )
        )
        roll_stats_plots[stat_type] = roll_stat_line
    return (
        ts + roll_stats_plots["mean"] + roll_stats_plots["stdev"]
    ).properties(width=fig_size[0], height=fig_size[1])


def manual_histogram(
    df: pd.DataFrame,
    yvar: str,
    hist_bin: alt.Bin = alt.Bin(step=0.5),
    bar_line_thickness: int = 2,
    bar_line_color: str = "blue",
    line_thickness: int = 5,
    axis_thickness: int = 2,
    figsize: tuple = (300, 300),
) -> alt.Chart:
    base = alt.Chart(df, title="Histogram")
    bar = base.mark_bar(
        strokeWidth=bar_line_thickness, color=bar_line_color
    ).encode(
        x=alt.X(
            f"{yvar}:Q",
            bin=hist_bin,
            axis=alt.Axis(
                title="",
                domainWidth=axis_thickness,
                domainColor="black",
                domainOpacity=1,
            ),
        ),
        y=alt.Y(
            "count()",
            axis=alt.Axis(
                title="",
                domainWidth=axis_thickness,
                domainColor="black",
                domainOpacity=1,
            ),
        ),
        tooltip=alt.Tooltip(yvar, bin=hist_bin),
    )
    rule = base.mark_rule(color="red").encode(
        x=f"mean({yvar}):Q", size=alt.value(line_thickness)
    )
    return (bar + rule).properties(width=figsize[0], height=figsize[1])


def ts_plot(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    ts_tooltip: list,
    corr_plots_tooltip: dict,
    line_roll_window: int = 5,
    line_linewidth=1.5,
    line_roll_stat_linewidth=3,
    line_roll_stat_colors: list = ["red", "black"],
    hist_bin: alt.Bin = alt.Bin(step=0.5),
    hist_bar_line_thickness=2,
    hist_bar_line_color="blue",
    hist_line_thickness=2,
    n_lags: int = 10,
    alpha: float = 0.05,
    ci_band_opacity: float = 0.5,
    corr_plots_wanted: list = ["acf", "pacf"],
    corr_plots_marker_size: int = 80,
    corr_plots_vertical_line_width: int = 2,
    corr_plots_marker_linewidth: int = 2,
    corr_plots_zero_y_linewidth: int = 2,
    axis_thickness: int = 2,
    line_plot_fig_size: tuple = (325, 275),
    hist_plot_fig_size: tuple = (325, 275),
    corr_plots_fig_size: tuple = (325, 275),
) -> alt.Chart:
    ts = plot_ts(
        df.reset_index(),
        xvar,
        yvar,
        ts_tooltip,
        line_roll_window,
        line_linewidth,
        line_roll_stat_linewidth,
        line_roll_stat_colors,
        line_plot_fig_size,
    )
    hist_plot = manual_histogram(
        df,
        yvar,
        hist_bin,
        hist_bar_line_thickness,
        hist_bar_line_color,
        hist_line_thickness,
        axis_thickness,
        hist_plot_fig_size,
    )
    corr_plots = plot_acf_pacf(
        df,
        yvar,
        n_lags,
        corr_plots_tooltip,
        alpha,
        ci_band_opacity,
        corr_plots_wanted,
        corr_plots_marker_size,
        corr_plots_vertical_line_width,
        corr_plots_marker_linewidth,
        corr_plots_zero_y_linewidth,
        corr_plots_fig_size,
    )
    ts_plot_obj = (
        alt.vconcat(
            alt.hconcat(ts, hist_plot).resolve_scale(color="independent"),
            corr_plots,
        )
        .configure_title(anchor="start", dx=35, offset=-3, fontSize=14)
        .configure_axis(labelFontSize=12, titleFontSize=14)
    )
    return ts_plot_obj
