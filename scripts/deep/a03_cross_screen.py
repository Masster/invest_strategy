"""Stage 1b: межинструментальный скрининг (lead-lag и пары) на периоде разработки -> data/cache/screen/x1."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import data as D  # noqa: E402
from moexlab.deep import xasset as X  # noqa: E402


def w(job):
    try:
        if job[0] == "LL":
            return X.run_leadlag_job(job[1], job[2])
        return X.run_pair_job(job[1], job[2], job[3])
    except Exception:  # noqa: BLE001
        return {"job": job, "error": traceback.format_exc()[-800:]}


if __name__ == "__main__":
    jobs = [("LL", c, tf) for tf in (5, 15, 60) for c in D.ALL_CODES]
    jobs += [("PAIR", a, b, tf) for tf in (5, 15, 60, 1440) for a, b in X.PAIRS]
    with mp.get_context("fork").Pool(4) as pool:
        for r in pool.imap_unordered(w, jobs):
            print(json.dumps(r, default=str)[:600], flush=True)
