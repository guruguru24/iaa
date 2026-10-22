"""Are late-reported claims more expensive? (motor bodily_injury and buildings subsidence)

If yes, count-based IBNR (which assumes late claims cost the same as reported ones) understates recent years.
Compares, per peril and loss year, the count-based factor with a cost-based factor that uses the reporting-delay
CDF weighted by net cost from mature loss years 2021-22.

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\check_bi_severity.py
"""
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data import load_claims

pd.set_option("display.width", 220)
c = load_claims("w1")
c = c[c.status != "repudiated"].copy()
c["net"] = (c.incurred_to_date - c.recoveries).clip(lower=0)
c["lag"] = (c.report_date - c.loss_date).dt.days
c["loss_year"] = c.loss_date.dt.year
cutoff = c.report_date.max()
c["window"] = (cutoff - c.loss_date).dt.days
MATURE = [2021, 2022]


def weighted_cdf(lags, weights):
    order = np.argsort(lags)
    l, w = np.asarray(lags)[order], np.asarray(weights, float)[order]
    cw = np.cumsum(w) / w.sum()
    return lambda x: np.interp(x, l, cw, left=0.0, right=1.0)


for line, peril in (("motor", "bodily_injury"), ("buildings", "subsidence")):
    p = c[(c.line == line) & (c.peril == peril)]
    m = p[p.loss_year.isin(MATURE)]
    print(f"\n{'=' * 20} {line} / {peril}")
    bins = [-1, 30, 90, 180, 365, 730, 10_000]
    m = m.assign(lag_band=pd.cut(m.lag, bins))
    t = m.groupby("lag_band", observed=True).agg(n=("claim_id", "size"), mean_net=("net", "mean"),
                                                 median_net=("net", "median"),
                                                 mean_capped_100k=("net", lambda s: s.clip(upper=100_000).mean()),
                                                 share_open=("status", lambda s: (s == "open").mean()))
    t["share_of_cost"] = m.groupby("lag_band", observed=True).net.sum() / m.net.sum()
    print("--- 1. Severity by reporting delay, loss years 2021-22"); print(t.round(3).to_string())
    Fn = weighted_cdf(m.lag, np.ones(len(m)))
    Fc = weighted_cdf(m.lag, m.net)
    out = []
    for y, g in p.groupby("loss_year"):
        pn = np.clip(Fn(g.window), 1e-3, 1)
        pc = np.clip(Fc(g.window), 1e-3, 1)
        out.append({"loss_year": y, "reported_n": len(g), "reported_net": g.net.sum(),
                    "count_factor": (1 / pn).sum() / len(g),
                    # per loss date: reported cost / share of cost reported by that delay = estimated ultimate cost
                    "cost_factor": (g.net / pc).sum() / g.net.sum()})
    print("--- 2. Count-based vs cost-based development factor by loss year")
    print(pd.DataFrame(out).round(4).to_string(index=False))
    sev = p.groupby("loss_year").agg(n=("claim_id", "size"), mean_net=("net", "mean"),
                                     mean_capped_100k=("net", lambda s: s.clip(upper=100_000).mean()),
                                     share_open=("status", lambda s: (s == "open").mean()),
                                     median_lag=("lag", "median"))
    print("--- 3. Reported severity by loss year"); print(sev.round(3).to_string())
