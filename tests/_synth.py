"""Synthetic inputs for tests. No test reads the real snapshot under data/."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from devtrial import frozen

CFG = frozen.load_config()


def modules():
    return frozen.load_modules(CFG)


def target_days(mp, start: dt.date, end: dt.date):
    out, d = [], start
    while d <= end:
        if mp.is_target_publication_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def allow_rows(mp, start=dt.date(2014, 1, 2), end=dt.date(2018, 6, 29), seed=7, gaps=(), flat=None):
    """(date, jpy, usd) rows: USD/JPY is a random walk with a weak AR(1) momentum component."""
    rng = np.random.Generator(np.random.PCG64(seed))
    days = target_days(mp, start, end)
    usd = 1.2
    lp = math.log(110.0)
    drift = 0.0
    rows = []
    for d in days:
        drift = 0.97 * drift + 0.0006 * rng.standard_normal()
        step = drift + 0.006 * rng.standard_normal()
        if flat is None or not (flat[0] <= d <= flat[1]):
            lp += step  # inside `flat` the price is held constant (F_MOM == 0 exactly after 20 days)
        jpy = usd * math.exp(lp)
        if d in gaps:
            rows.append((d, None, usd))
        else:
            rows.append((d, jpy, usd))
    return rows


def cfg(**over):
    c = dict(CFG)
    c.update(over)
    return c
