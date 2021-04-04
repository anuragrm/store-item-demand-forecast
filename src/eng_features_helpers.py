#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def calc_circ_dt_feat(trig_function, y_frac):
    return trig_function(2 * np.pi * y_frac)


def add_datepart(df, fldname, drop=True, inplace=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname].dt
    doy_fld = "dayofyear"
    doy = fld.dayofyear
    woy = fld.isocalendar().week
    for trcol, trig_func in zip([np.sin, np.cos], ["sin", "cos"]):
        df[f"{trig_func}_weekday"] = calc_circ_dt_feat(trcol, fld.weekday / 7)
        df[f"{trig_func}_month"] = calc_circ_dt_feat(trcol, fld.month / 12)
        df[f"{trig_func}_{doy_fld}"] = calc_circ_dt_feat(trcol, doy / 365)
        df[f"{trig_func}_quarter"] = calc_circ_dt_feat(trcol, fld.quarter / 4)
        df[f"{trig_func}_woy"] = calc_circ_dt_feat(trcol, woy / 52)
        df[f"{trig_func}_woy"] = df[f"{trig_func}_woy"].astype(float)
    if drop:
        df.drop(fldname, axis=1, inplace=True)
    if not inplace:
        return df
