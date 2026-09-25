"""Библиотека дневных портфельных стратегий (сетки параметров заданы заранее, до просмотра holdout)."""
from __future__ import annotations

import pandas as pd

from . import panel as PN


def library(P):
    L = []
    for cls in ("FUT", "EQ", None):
        cn = cls or "ALL"
        for vn in (20, 60):
            for n in (5, 10, 20, 40, 60, 120):
                L.append(("P_TSMOM", cn, dict(n=n, vn=vn), lambda n=n, vn=vn, cls=cls: PN.w_tsmom(P, n, vn=vn, cls=cls)))
            for f, s in ((3, 10), (5, 20), (10, 50), (20, 100), (50, 200)):
                L.append(("P_EMA", cn, dict(f=f, s=s, vn=vn), lambda f=f, s=s, vn=vn, cls=cls: PN.w_ema(P, f, s, vn=vn, cls=cls)))
            for n in (5, 10, 20, 55):
                L.append(("P_BRK", cn, dict(n=n, vn=vn), lambda n=n, vn=vn, cls=cls: PN.w_breakout(P, n, vn=vn, cls=cls)))
            L.append(("P_BUYHOLD", cn, dict(vn=vn), lambda vn=vn, cls=cls: PN.w_buyhold(P, vn=vn, cls=cls)))
    for cls, ks in (("FUT", (1, 2, 3)), ("EQ", (2, 3, 4))):
        for n in (1, 3, 5, 10, 20, 60):
            for k in ks:
                for rev in (False, True):
                    for hold in (1, 5):
                        L.append(("P_XS_REV" if rev else "P_XS_MOM", cls, dict(n=n, k=k, hold=hold),
                                  lambda n=n, k=k, rev=rev, hold=hold, cls=cls: PN.w_xs(P, n, k, cls=cls, reverse=rev, hold=hold)))
    L.append(("P_CARRY", "FX", dict(), lambda: PN.w_carry(P)))
    for f, s in ((10, 50), (20, 100)):
        def carry_tf(f=f, s=s):
            w = PN.w_carry(P)
            ef = P["c"].ewm(span=f, adjust=False).mean()
            es = P["c"].ewm(span=s, adjust=False).mean()
            return w.where(ef < es, 0.0)
        L.append(("P_CARRY_TF", "FX", dict(f=f, s=s), carry_tf))
    # только лонг / только шорт варианты трендов
    base = list(L)
    for fam, cn, p, fn in base:
        if fam in ("P_TSMOM", "P_EMA", "P_BRK"):
            L.append((fam + "_L", cn, p, lambda fn=fn: fn().clip(lower=0)))
            L.append((fam + "_S", cn, p, lambda fn=fn: fn().clip(upper=0)))
    return L


