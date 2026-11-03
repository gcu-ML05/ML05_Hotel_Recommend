"""
Step 4: deviation-based user preferences and rebuilt traveler personas

Background
    02_user_aspect_weights.py built aspect importance from per-user regression coefficients (Ridge shrunk
    toward the global weights). But users have only 3.7 complete reviews on average and the 6 aspects are
    highly correlated, so at the selected λ=1000 the personal weights were almost the global weights and
    the importance shares of the 3 personas differed by only about 0.003.

Idea (option (c) of the team discussion)
    Instead of regression coefficients (7 parameters to estimate), directly average how much the score a
    user gave each aspect differs from "that hotel's average aspect score". An average, not a coefficient
    estimate, stays stable even with 3-4 reviews.

    (1) raw deviation      d_uk = mean_i [ score_{u,i,k} - hotel_mean_{h(i),k} ]   (train reviews that have aspect k)
    (2) leniency           L_u  = mean of all of user u's aspect deviations          (tendency to rate everything high/low)
    (3) relative deviation c_uk = d_uk - L_u                                         (per-aspect strictness after removing leniency)
    (4) shrinkage          c_uk ← c_uk · n_uk / (n_uk + M)                            (aspects with few reviews pulled toward 0)

    c_uk < 0 : rates that aspect lower than others do = strict about it (read as an aspect the user cares about)
    c_uk > 0 : lenient on that aspect = cares less about it

    Clustering "strictness" = -c_uk with K-means yields personas such as "strict on cleanliness" or
    "strict on value".

Validation (is this representation a stable user trait?)
    Predict the aspect scores of val reviews and compare RMSE (val reviews come after the user's train
    reviews, so there is no time leakage).
        P0  hotel mean                                    hotel_mean_hk
        P1  + leniency                                    hotel_mean_hk + L_u
        P2  + leniency + persona centroid deviation       hotel_mean_hk + L_u + centroid_{p(u),k}
        P3  + leniency + personal relative deviation      hotel_mean_hk + L_u + c_uk
    P3 < P1 means per-aspect personal tendencies exist beyond leniency;
    P2 better than P1 means the personas capture at least part of them.

Input:  data/pipeline/ratings_split.csv   (01_split.py)
        data/pipeline/hotel_aspects.npy   (03_hotel_features.py, Bayesian hotel aspect means from train reviews only)
Output: data/pipeline/user_personas_dev.csv, persona_dev_centroids.csv
        Pipeline/outputs/04_*.csv, figures/04_*.png

Run:
      cd Pipeline && python 04_persona_deviation.py          (K chosen automatically)
      cd Pipeline && python 04_persona_deviation.py --k 4    (K given)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import ASPECTS, SEED, WORK_DIR, ensure_dirs, load_split, rmse, save_fig, save_table, setup_plot

M = 3                 # shrinkage strength: with M reviews for an aspect the deviation is halved (pseudo-reviews)
MIN_PER_ASPECT = 3    # users used for persona fitting: at least this many train reviews for all 6 aspects
K_RANGE = range(2, 9)


def user_deviations(tr, A, n_users):
    """
    Per-user aspect deviation statistics
    Returns: dev_sum (n_users, 6), cnt (n_users, 6)
             dev_sum[u, k] = Σ (user score - hotel mean),  cnt[u, k] = number of reviews with that aspect
    """
    K = len(ASPECTS)
    dev_sum = np.zeros((n_users, K))
    cnt = np.zeros((n_users, K))
    for k, a in enumerate(ASPECTS):
        s = tr[["user_id", "hotel_id", a]].dropna()
        u = s.user_id.to_numpy()
        r = s[a].to_numpy(float) - A[s.hotel_id.to_numpy(), k]
        # np.bincount(indices, weights=values, minlength=length): array of summed values per index
        dev_sum[:, k] = np.bincount(u, weights=r, minlength=n_users)
        cnt[:, k] = np.bincount(u, minlength=n_users)
    return dev_sum, cnt


def build_profiles(dev_sum, cnt):
    """
    Build leniency L_u (shrunk) and relative deviation c_uk (shrunk).
    Leniency is computed over all aspect deviations together and shrunk by the total number of reviews.
    """
    n_all = cnt.sum(1)
    L_raw = dev_sum.sum(1) / np.maximum(n_all, 1)
    L = L_raw * n_all / (n_all + M)                              # 0 when there are no reviews (= an average user)
    d = np.where(cnt > 0, dev_sum / np.maximum(cnt, 1), 0.0)      # raw deviation per aspect
    c = np.where(cnt > 0, d - L_raw[:, None], 0.0)                # relative deviation after removing leniency
    c = c * cnt / (cnt + M)
    return L, c


def val_rmse_table(va, A, L, c, persona, cent_c):
    """RMSE of predicting the aspect scores of val reviews (per aspect + overall)"""
    rows, pooled = [], {p: ([], []) for p in ["P0", "P1", "P2", "P3"]}
    for k, a in enumerate(ASPECTS):
        s = va[["user_id", "hotel_id", a]].dropna()
        u, h, y = s.user_id.to_numpy(), s.hotel_id.to_numpy(), s[a].to_numpy(float)
        base = A[h, k]
        preds = {
            "P0": base,
            "P1": base + L[u],
            "P2": base + L[u] + cent_c[persona[u], k],
            "P3": base + L[u] + c[u, k],
        }
        row = {"aspect": a, "n_val": len(y)}
        for p, pr in preds.items():
            pr = np.clip(pr, 1, 5)
            row[p] = rmse(y, pr)
            pooled[p][0].append(y)
            pooled[p][1].append(pr)
        rows.append(row)
    total = {"aspect": "ALL", "n_val": sum(r["n_val"] for r in rows)}
    for p, (ys, ps) in pooled.items():
        total[p] = rmse(np.concatenate(ys), np.concatenate(ps))
    tab = pd.DataFrame(rows + [total])
    tab = tab.rename(columns={"P0": "P0_hotel_mean", "P1": "P1_+leniency",
                              "P2": "P2_+leniency+persona", "P3": "P3_+leniency+personal"})
    return tab


def main():
    """profiles → choose K and cluster → save personas → val check → save figures"""
    # argparse.ArgumentParser: command-line parser, add_argument(name, type=type, default=default, help=description)
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="number of personas (if omitted, chosen by silhouette score)")
    args = ap.parse_args()
    ensure_dirs()
    plt = setup_plot()

    df, n_users, n_hotels = load_split()
    A = np.load(WORK_DIR / "hotel_aspects.npy")
    assert A.shape == (n_hotels, len(ASPECTS)), f"hotel_aspects.npy shape {A.shape} does not match {n_hotels} hotels"
    tr, va = df[df.split == "train"], df[df.split == "val"]

    # (1)-(4) user profiles
    dev_sum, cnt = user_deviations(tr, A, n_users)
    L, c = build_profiles(dev_sum, cnt)
    strict = -c                                                     # positive = strict about that aspect
    print(f"leniency L_u: mean {L.mean():+.3f}, std {L.std():.3f}")
    print("relative deviation std (per aspect):", dict(zip(ASPECTS, c.std(0).round(3))))

    # users for persona fitting
    mask = (cnt >= MIN_PER_ASPECT).all(1)
    S = strict[mask]
    print(f"users for persona fitting ({MIN_PER_ASPECT}+ reviews for all 6 aspects): {mask.sum():,} / {n_users:,}")

    # choose K: silhouette score + spread of centroids (about 0.003 with the previous method)
    rng = np.random.default_rng(SEED)
    sample = S[rng.choice(len(S), size=min(20_000, len(S)), replace=False)]
    rows = []
    for k in K_RANGE:
        # KMeans(n_clusters=number of clusters, n_init=best of several initializations, random_state=reproducibility)
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
        spread = np.abs(km.cluster_centers_ - S.mean(0)).max()      # how far the farthest centroid is from the overall mean
        # silhouette_score(data, labels): -1 to 1, higher = better separated (20k-user sample to limit cost)
        rows.append({"k": k, "silhouette": silhouette_score(sample, km.predict(sample)),
                     "inertia": km.inertia_, "max_centroid_dev": spread})
    ksel = pd.DataFrame(rows)
    print(ksel.round(4).to_string(index=False))
    save_table(ksel, "04_kmeans_k_selection")
    k = args.k or int(ksel.loc[ksel.silhouette.idxmax(), "k"])
    print(f"selected K = {k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
    persona = km.predict(strict)                                    # assign every user to the nearest cluster
    cent = pd.DataFrame(km.cluster_centers_, columns=ASPECTS)
    # name: the aspect the persona is strictest about (largest centroid strictness)
    cent["name"] = [f"P{i}_strict_{ASPECTS[int(np.argmax(r))]}" for i, r in enumerate(km.cluster_centers_)]
    cent["n_users_fit"] = np.bincount(km.labels_, minlength=k)
    cent["n_users_all"] = np.bincount(persona, minlength=k)
    cent.insert(0, "persona", range(k))
    print(cent.round(3).to_string(index=False))
    cent.to_csv(WORK_DIR / "persona_dev_centroids.csv", index=False)
    save_table(cent.round(4), "04_persona_dev_centroids")

    # also save an importance-share version (same format as 02, sums to 1) to plug into 08_kaggle_generalization.py etc.
    # tilt the global weight shares by strictness: share_k ∝ w_g,k · exp(strictness_k)
    w_path = WORK_DIR / "global_weights.npy"
    if w_path.exists():
        w_g = np.clip(np.load(w_path)[1:], 1e-6, None)
        tilt = w_g * np.exp(km.cluster_centers_)
        share = tilt / tilt.sum(1, keepdims=True)
        share_tab = pd.DataFrame(share, columns=ASPECTS)
        share_tab.insert(0, "persona", range(k))
        share_tab["name"] = cent["name"]
        save_table(share_tab.round(4), "04_persona_dev_share")

    users = pd.DataFrame(strict, columns=[f"strict_{a}" for a in ASPECTS])
    users.insert(0, "leniency", L)
    users.insert(0, "persona", persona)
    users.insert(0, "n_train_aspect_reviews", cnt.max(1).astype(int))
    users.insert(0, "user_id", np.arange(n_users))
    users.to_csv(WORK_DIR / "user_personas_dev.csv", index=False)

    # validation: predict val aspect scores
    cent_c = -km.cluster_centers_                                   # centroids back in relative-deviation (c) units
    tab = val_rmse_table(va, A, L, c, persona, cent_c)
    print(tab.round(4).to_string(index=False))
    save_table(tab.round(5), "04_val_aspect_rmse")

    # figure 1: choosing K
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ksel.k, ksel.silhouette, marker="o", color="C1")
    ax[0].axvline(k, ls="--", color="gray")
    ax[0].set_title("Silhouette score")
    ax[1].plot(ksel.k, ksel.max_centroid_dev, marker="o")
    ax[1].set_title("Max distance of a centroid from the overall mean (score units)")
    for a in ax:
        a.set_xlabel("K")
    save_fig(plt, "04_kmeans_k_selection")

    # figure 2: strictness per persona
    C = cent[ASPECTS].to_numpy()
    lim = max(np.abs(C).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 0.6 * k + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(ASPECTS)), ASPECTS, rotation=30)
    ax.set_yticks(range(k), cent["name"] + " (" + cent.n_users_fit.astype(str) + ")")
    for i in range(k):
        for j in range(len(ASPECTS)):
            ax.text(j, i, f"{C[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Aspect strictness per persona (+: rates lower than the hotel average)")
    plt.colorbar(im)
    save_fig(plt, "04_persona_dev_strictness")


if __name__ == "__main__":
    main()
