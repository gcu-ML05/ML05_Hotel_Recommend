"""
Shared settings and helpers for the pipeline

Paths
    data/processing/hotelrec/   preprocessing output (Preprocessing/01)
    data/processing/kaggle/     preprocessing output (Preprocessing/02)
    data/pipeline/              intermediate pipeline artifacts (split, weights, hotel features / not in Git)
    Pipeline/outputs/           result tables (csv) and figures (png) (in Git)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HOTELREC_DIR = ROOT / "data" / "processing" / "hotelrec"
KAGGLE_DIR = ROOT / "data" / "processing" / "kaggle"
WORK_DIR = ROOT / "data" / "pipeline"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

ASPECTS = ["service", "cleanliness", "value", "rooms", "location", "sleep_quality"]
SEED = 42


def ensure_dirs():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def load_ratings():
    """Load ratings only from train (all sub-ratings) + partial (some/none) combined (text skipped to save memory)"""
    cols = ["user_id", "hotel_id", "rating", "date", *ASPECTS]
    parts = [pd.read_csv(HOTELREC_DIR / f, usecols=cols, encoding="utf-8-sig")
             for f in ("hotelrec_train.csv", "hotelrec_partial.csv")]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_split():
    """Output of 01_split.py (split column: train / val / test)"""
    df = pd.read_csv(WORK_DIR / "ratings_split.csv", parse_dates=["date"])
    n_users, n_hotels = df.user_id.max() + 1, df.hotel_id.max() + 1
    return df, n_users, n_hotels


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def sample_candidates(df_all, users, pos_hotels, n_hotels, n_neg, rng):
    """
    Candidates for Top-N evaluation: 1 ground-truth hotel + n_neg hotels the user never rated
    Returns: array of shape (len(users), 1 + n_neg); column 0 is the ground truth
    """
    rated = df_all.groupby("user_id").hotel_id.apply(set)
    cand = np.empty((len(users), 1 + n_neg), dtype=np.int64)
    cand[:, 0] = pos_hotels
    for i, u in enumerate(users):
        seen = rated[u]
        negs = []
        while len(negs) < n_neg:
            for h in rng.integers(0, n_hotels, size=2 * n_neg):
                if h not in seen and h not in negs:
                    negs.append(h)
                    if len(negs) == n_neg:
                        break
        cand[i, 1:] = negs
    return cand


def sample_city_candidates(df_all, users, pos_hotels, hotel_city, n_neg, rng):
    """
    Same-city candidates: n_neg hotels in the same city as the ground truth (hotel_map location) that the user never rated.
    Evaluates "which hotel does a user pick once the destination is decided".
    Users whose city has fewer than n_neg candidates are excluded.
    Returns: (array of evaluated users, candidate array of shape (n_users, 1 + n_neg))
    """
    rated = df_all.groupby("user_id").hotel_id.apply(set)
    city_hotels = pd.Series(np.arange(len(hotel_city))).groupby(hotel_city).apply(np.array)
    keep, cands = [], []
    for u, h in zip(users, pos_hotels):
        pool = city_hotels[hotel_city[h]]
        pool = pool[~np.isin(pool, list(rated[u]))]
        if len(pool) < n_neg:
            continue
        keep.append(u)
        cands.append(np.r_[h, rng.choice(pool, size=n_neg, replace=False)])
    return np.array(keep), np.array(cands)


def ranking_metrics(scores, ks=(5, 10)):
    """
    scores: (n_users, 1 + n_neg); column 0 is the ground-truth hotel's score
    Rank of the ground truth = number of candidates scored higher + half the number of ties
    (so that models producing many ties are not favored)
    HR@k: 1 if the ground truth is in the top k / NDCG@k: 1/log2(rank+2)
    """
    pos = scores[:, :1]
    rank = (scores[:, 1:] > pos).sum(1) + 0.5 * (scores[:, 1:] == pos).sum(1)
    out = {}
    for k in ks:
        hit = rank < k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1 / np.log2(rank + 2), 0).mean())
    return out, rank


def setup_plot():
    """matplotlib setup (uses a Korean-capable font if installed, as in the EDA notebook)"""
    import matplotlib
    matplotlib.use("Agg")                      # save to files only, no display
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if f in installed:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save_fig(plt, name):
    path = OUT_DIR / "figures" / f"{name}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   saved figure: {path.relative_to(ROOT)}")


def save_table(df, name, index=False):
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"   saved table: {path.relative_to(ROOT)}")
