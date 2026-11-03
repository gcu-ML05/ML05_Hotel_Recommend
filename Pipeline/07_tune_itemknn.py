"""
Step 7 (optional): Item-kNN hyperparameter selection (val)

Compares the original version (centered on the user mean, adjusted cosine) with the improved version
(KNNBaseline) on the val split, and records why 06_train_evaluate.py uses its defaults (k, shrink, damp).

    val RMSE              : rating prediction for every user's val review
    HR@10 (predict)       : ranking by the predicted rating
    HR@10 (rank_score)    : ranking by the ranking score (sum of co-visit similarities)
    neighbor_coverage     : share of val pairs where the user has rated at least one neighbor
    (Top-N uses 5,000 val users, 1 ground truth + 99 random candidates)

Each setting takes about 4-5 minutes to compute similarities (about 30 minutes in total).
Output: Pipeline/outputs/07_itemknn_tuning.csv
"""
import time

import numpy as np
import pandas as pd

from common import SEED, ensure_dirs, load_split, ranking_metrics, rmse, sample_candidates, save_table
from models import BiasBaseline, ItemKNN

# (center, shrink, k, list of damp values)
CONFIGS = [
    ("user_mean", 10, 50, [0]),                  # original version
    ("baseline", 100, 50, [0, 0.03, 0.3, 1]),
    ("baseline", 100, 200, [0.03]),
    ("baseline", 100, 500, [0.03]),
    ("baseline", 100, 2000, [0.03]),
]


def main():
    ensure_dirs()
    df, n_users, n_hotels = load_split()
    tr, va = df[df.split == "train"], df[df.split == "val"]
    u, i, r = tr.user_id.to_numpy(), tr.hotel_id.to_numpy(), tr.rating.to_numpy(float)
    vu, vi, vr = va.user_id.to_numpy(), va.hotel_id.to_numpy(), va.rating.to_numpy(float)

    rng = np.random.default_rng(SEED)
    users = np.sort(rng.choice(va.user_id.unique(), 5000, replace=False))
    pos = va.set_index("user_id").loc[users, "hotel_id"].to_numpy()
    cand = sample_candidates(df, users, pos, n_hotels, 99, rng)
    cu, ci = np.repeat(users, cand.shape[1]), cand.ravel()

    base = BiasBaseline().fit(u, i, r, n_users, n_hotels)
    rows = [{"model": "Baseline(μ+b_u+b_i)", "val_RMSE": rmse(vr, base.predict(vu, vi)),
             "HR@10_predict": ranking_metrics(base.predict(cu, ci).reshape(cand.shape))[0]["HR@10"]}]
    for center, shrink, k, damps in CONFIGS:
        t0 = time.time()
        m = ItemKNN(k=k, shrink=shrink, center=center).fit(u, i, r, n_users, n_hotels)
        den = np.asarray((m.B[vu] @ m.ST)[np.arange(len(vu)), vi]).ravel()
        hr_rank = ranking_metrics(m.rank_score(cu, ci).reshape(cand.shape))[0]["HR@10"]
        for d in damps:
            m.damp = d
            rows.append({"model": f"ItemKNN center={center} shrink={shrink} k={k} damp={d}",
                         "val_RMSE": rmse(vr, m.predict(vu, vi)),
                         "HR@10_predict": ranking_metrics(m.predict(cu, ci).reshape(cand.shape))[0]["HR@10"],
                         "HR@10_rank_score": hr_rank, "neighbor_coverage": float((den > 0).mean()),
                         "fit_s": round(time.time() - t0)})
            print(rows[-1], flush=True)
    tab = pd.DataFrame(rows)
    print(tab.round(4).to_string(index=False))
    save_table(tab, "07_itemknn_tuning")


if __name__ == "__main__":
    main()
