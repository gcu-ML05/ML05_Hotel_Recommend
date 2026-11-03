"""
Step 5: improving the deviation-based personas (04) and the final test evaluation

Open issues from 04
    (1) The shrinkage strength M=3 was chosen without evidence. P3 (personal deviation) was worse than
        P1 (leniency only), possibly because weak shrinkage let noise in.
    (2) Leniency L_u was added to every aspect with the same size. For location, P1 was worse than
        P0 (hotel mean), a sign that leniency carries over differently per aspect.
    (3) K was chosen by silhouette only (K=2). K=3 and 4 should also be compared for interpretability
        and predictive power.
    (4) Hyperparameters were chosen on val and reported on val. Final numbers must be reported on test.

What this script does
    A. Choose M (val):         c_uk ← c_uk · n_uk / (n_uk + M), the M in M_GRID with the lowest val P3 RMSE
    B. Per-aspect leniency coefficient (estimated on val):
           β_k = argmin_β Σ (y - hotel_mean_k - β · L_u)²  =  Σ L_u (y - hotel_mean_k) / Σ L_u²   (closed-form least squares)
           P1β = hotel_mean_k + β_k · L_u
    C. Compare K (val):        K in K_GRID; silhouette, centroid spread, P2 RMSE, smallest cluster share
    D. Final test evaluation:  fix the settings from A-C and report P0-P3 RMSE on test reviews (profiles use train only)
    E. Persona summary table:  readable name, size, mean leniency, mean rating, mean number of reviews

Input:  data/pipeline/ratings_split.csv, hotel_aspects.npy (same as 04)
Output: Pipeline/outputs/05_*.csv, figures/05_*.png
        data/pipeline/user_personas_dev_tuned.csv (format usable by 06 training/evaluation, with the selected M and K)
Run:    cd Pipeline && python 05_persona_tuning.py
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import ASPECTS, SEED, WORK_DIR, ensure_dirs, load_split, rmse, save_fig, save_table, setup_plot

# the file name starts with a digit, so a plain import does not work; reuse 04's functions via importlib.import_module(name)
dev04 = importlib.import_module("04_persona_deviation")

M_L = dev04.M                         # leniency shrinkage fixed as in 04 (L pools all 6 aspects, so it matters little)
M_GRID = [1, 3, 10, 30, 100, 300]     # candidate shrinkage strengths for the relative deviation (M=0 excluded: 0/0 for aspects with no reviews)
K_GRID = [2, 3, 4, 5]
K_TOL = 0.0005                        # if val RMSE is within this of the minimum, pick the smaller (easier to read) K
KR = {"service": "service", "cleanliness": "cleanliness", "value": "value",
      "rooms": "rooms", "location": "location", "sleep_quality": "sleep"}


def profiles(dev_sum, cnt, m_c):
    """
    Leniency L_u and relative deviation c_uk. Same as 04's build_profiles, except that the shrinkage of c is m_c.
        L_u  = (Σ_k dev_sum_uk / Σ_k n_uk) · n_u / (n_u + M_L)
        c_uk = (dev_sum_uk / n_uk - L_raw_u) · n_uk / (n_uk + m_c)
    """
    n_all = cnt.sum(1)
    L_raw = dev_sum.sum(1) / np.maximum(n_all, 1)
    L = L_raw * n_all / (n_all + M_L)
    d = np.where(cnt > 0, dev_sum / np.maximum(cnt, 1), 0.0)
    c = np.where(cnt > 0, d - L_raw[:, None], 0.0)
    return L, c * cnt / (cnt + m_c)


def aspect_arrays(df, A):
    """Per-aspect arrays of (user, hotel mean, actual score). Missing aspects are skipped."""
    out = {}
    for k, a in enumerate(ASPECTS):
        s = df[["user_id", "hotel_id", a]].dropna()
        out[a] = (s.user_id.to_numpy(), A[s.hotel_id.to_numpy(), k], s[a].to_numpy(float))
    return out


def rmse_table(arr, preds):
    """
    preds: {name: f(k, u, base) -> prediction}. Table of per-aspect RMSE and overall RMSE (ALL, pooled over all aspect scores).
    Predictions are clipped to the rating range [1, 5] (np.clip(value, min, max)).
    """
    rows, pooled = [], {p: ([], []) for p in preds}
    for k, a in enumerate(ASPECTS):
        u, base, y = arr[a]
        row = {"aspect": a, "n": len(y)}
        for p, f in preds.items():
            pr = np.clip(f(k, u, base), 1, 5)
            row[p] = rmse(y, pr)
            pooled[p][0].append(y)
            pooled[p][1].append(pr)
        rows.append(row)
    total = {"aspect": "ALL", "n": sum(r["n"] for r in rows)}
    for p, (ys, ps) in pooled.items():
        total[p] = rmse(np.concatenate(ys), np.concatenate(ps))
    return pd.DataFrame(rows + [total])


def fit_beta(arr, L):
    """Per-aspect leniency coefficient β_k = Σ L_u (y - base) / Σ L_u²  (least squares of a regression without intercept)"""
    beta = np.ones(len(ASPECTS))
    for k, a in enumerate(ASPECTS):
        u, base, y = arr[a]
        beta[k] = (L[u] * (y - base)).sum() / max((L[u] ** 2).sum(), 1e-12)
    return beta


def persona_name(center):
    """Centroid (strictness) → readable name: the strictest aspect + the most lenient aspect."""
    hi, lo = ASPECTS[int(np.argmax(center))], ASPECTS[int(np.argmin(center))]
    return f"Strict on {KR[hi]} (lenient on {KR[lo]})"


def fit_personas(strict, mask, k):
    """Fit K-means on users with enough reviews for all 6 aspects, then assign everyone. Returns: (model, assignments)"""
    # KMeans(n_clusters=number of clusters, n_init=best of several initializations, random_state=reproducibility)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(strict[mask])
    return km, km.predict(strict)


def main():
    """Run experiments A-E in order and save tables and figures."""
    ensure_dirs()
    plt = setup_plot()
    df, n_users, n_hotels = load_split()
    A = np.load(WORK_DIR / "hotel_aspects.npy")
    tr = df[df.split == "train"]
    va_arr = aspect_arrays(df[df.split == "val"], A)
    te_arr = aspect_arrays(df[df.split == "test"], A)

    dev_sum, cnt = dev04.user_deviations(tr, A, n_users)
    mask = (cnt >= dev04.MIN_PER_ASPECT).all(1)

    # ---------------- A. choose shrinkage strength M (val) ----------------
    rows = []
    for m in M_GRID:
        L, c = profiles(dev_sum, cnt, m)
        t = rmse_table(va_arr, {"P1": lambda k, u, b: b + L[u],
                                "P3": lambda k, u, b: b + L[u] + c[u, k]})
        r = t.set_index("aspect")
        rows.append({"M": m, "val_P1": r.loc["ALL", "P1"], "val_P3": r.loc["ALL", "P3"],
                     "val_P3_location": r.loc["location", "P3"], "P3_minus_P1": r.loc["ALL", "P3"] - r.loc["ALL", "P1"]})
    mtab = pd.DataFrame(rows)
    print(mtab.round(5).to_string(index=False))
    save_table(mtab.round(5), "05_shrinkage_M")
    m_best = int(mtab.loc[mtab.val_P3.idxmin(), "M"])
    print(f"selected M = {m_best}")
    L, c = profiles(dev_sum, cnt, m_best)
    strict = -c

    # ---------------- B. per-aspect leniency coefficient β (val) ----------------
    beta = fit_beta(va_arr, L)
    btab = pd.DataFrame({"aspect": ASPECTS, "beta": beta})
    print(btab.round(3).to_string(index=False))
    save_table(btab.round(4), "05_leniency_beta")

    # ---------------- C. compare K (val) ----------------
    rng = np.random.default_rng(SEED)
    idx = rng.choice(mask.sum(), size=min(20_000, mask.sum()), replace=False)
    rows, models = [], {}
    for k in K_GRID:
        km, persona = fit_personas(strict, mask, k)
        models[k] = (km, persona)
        cc = -km.cluster_centers_                                   # centroids in relative-deviation (c) units
        t = rmse_table(va_arr, {"P2": lambda kk, u, b: b + beta[kk] * L[u] + cc[persona[u], kk]}).set_index("aspect")
        S = strict[mask]
        # silhouette_score(data, labels): -1 to 1, higher = better separated (computed on a 20k-user sample)
        rows.append({"k": k, "silhouette": silhouette_score(S[idx], km.predict(S[idx])),
                     "max_centroid_dev": np.abs(km.cluster_centers_ - S.mean(0)).max(),
                     "smallest_share": np.bincount(persona, minlength=k).min() / n_users,
                     "val_P2": t.loc["ALL", "P2"], "val_P2_location": t.loc["location", "P2"],
                     "names": " / ".join(persona_name(r) for r in km.cluster_centers_)})
    ktab = pd.DataFrame(rows)
    print(ktab.round(4).to_string(index=False))
    save_table(ktab.round(5), "05_k_comparison")
    # parsimony rule: the smallest K whose val P2 is within K_TOL of the minimum (a larger K for a tiny gain overfits val)
    k_best = int(ktab.loc[ktab.val_P2 <= ktab.val_P2.min() + K_TOL, "k"].min())
    print(f"selected K = {k_best} (smallest K within {K_TOL} of the minimum val P2 RMSE)")
    km, persona = models[k_best]
    cc = -km.cluster_centers_

    # share format to plug into step 10 (Kaggle comparison) (same transform as 04: share_k ∝ global weight_k · exp(strictness_k))
    w_g = np.clip(np.load(WORK_DIR / "global_weights.npy")[1:], 1e-6, None)
    tilt = w_g * np.exp(km.cluster_centers_)
    share = pd.DataFrame(tilt / tilt.sum(1, keepdims=True), columns=ASPECTS)
    share.insert(0, "persona", range(k_best))
    share["name"] = [f"T{i}_{persona_name(r)}" for i, r in enumerate(km.cluster_centers_)]
    share["n_users_fit"] = np.bincount(km.labels_, minlength=k_best)
    save_table(share.round(4), "05_persona_dev_share")

    # ---------------- D. final test evaluation ----------------
    L0, c0 = profiles(dev_sum, cnt, M_L)                             # original setting of 04 (M=3, β=1, K=2)
    km0, persona0 = fit_personas(-c0, (cnt >= dev04.MIN_PER_ASPECT).all(1), 2)
    cc0 = -km0.cluster_centers_
    preds = {
        "P0 hotel mean": lambda k, u, b: b,
        "P1 +leniency (04)": lambda k, u, b: b + L0[u],
        "P2 +persona (04: M=3, K=2)": lambda k, u, b: b + L0[u] + cc0[persona0[u], k],
        "P3 +personal deviation (04: M=3)": lambda k, u, b: b + L0[u] + c0[u, k],
        "P1β per-aspect leniency": lambda k, u, b: b + beta[k] * L[u],
        f"P2β +persona (M={m_best}, K={k_best})": lambda k, u, b: b + beta[k] * L[u] + cc[persona[u], k],
        f"P3β +personal deviation (M={m_best})": lambda k, u, b: b + beta[k] * L[u] + c[u, k],
    }
    test_tab = rmse_table(te_arr, preds)
    val_tab = rmse_table(va_arr, preds)
    print("test\n", test_tab.round(4).to_string(index=False))
    save_table(test_tab.round(5), "05_test_aspect_rmse")
    save_table(val_tab.round(5), "05_val_aspect_rmse")

    # ---------------- E. persona summary table ----------------
    rating_sum = np.bincount(tr.user_id, weights=tr.rating, minlength=n_users)
    rating_cnt = np.bincount(tr.user_id, minlength=n_users)
    summ = []
    for label, (kmx, pers, Lx) in {"04 original (M=3, K=2)": (km0, persona0, L0),
                                   f"improved (M={m_best}, K={k_best})": (km, persona, L)}.items():
        for p in range(kmx.n_clusters):
            sel = pers == p
            summ.append({"setting": label, "persona": p, "name": persona_name(kmx.cluster_centers_[p]),
                         "n_users": int(sel.sum()), "share_%": 100 * sel.mean(),
                         "mean_leniency": Lx[sel].mean(),
                         "mean_rating": rating_sum[sel].sum() / max(rating_cnt[sel].sum(), 1),
                         "mean_train_reviews": rating_cnt[sel].mean(),
                         **{f"strict_{a}": kmx.cluster_centers_[p, j] for j, a in enumerate(ASPECTS)}})
    summ = pd.DataFrame(summ)
    print(summ.round(3).to_string(index=False))
    save_table(summ.round(4), "05_persona_summary")

    # per-user persona file used by later steps (improved setting)
    users = pd.DataFrame(strict, columns=[f"strict_{a}" for a in ASPECTS])
    users.insert(0, "leniency", L)
    users.insert(0, "persona_name", [persona_name(r) for r in km.cluster_centers_[persona]])
    users.insert(0, "persona", persona)
    users.insert(0, "user_id", np.arange(n_users))
    users.to_csv(WORK_DIR / "user_personas_dev_tuned.csv", index=False)

    # figure: val RMSE by M
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = np.arange(len(M_GRID))
    ax.plot(xs, mtab.val_P1, marker="o", label="P1 +leniency")
    ax.plot(xs, mtab.val_P3, marker="o", label="P3 +leniency +personal deviation")
    ax.set_xticks(xs, [str(m) for m in M_GRID])
    ax.set_xlabel("Shrinkage strength M")
    ax.set_ylabel("val RMSE (all aspects)")
    ax.set_title("val RMSE by relative-deviation shrinkage")
    ax.legend()
    save_fig(plt, "05_shrinkage_M")

    # figure: strictness of the personas at the selected K
    C = km.cluster_centers_
    lim = max(np.abs(C).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 0.6 * k_best + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(ASPECTS)), [KR[a] for a in ASPECTS])
    n = np.bincount(persona, minlength=k_best)
    ax.set_yticks(range(k_best), [f"{persona_name(r)} ({n[i]:,} users)" for i, r in enumerate(C)])
    for i in range(k_best):
        for j in range(len(ASPECTS)):
            ax.text(j, i, f"{C[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Improved personas (M={m_best}, K={k_best}): aspect strictness")
    plt.colorbar(im)
    save_fig(plt, "05_persona_strictness")


if __name__ == "__main__":
    main()
