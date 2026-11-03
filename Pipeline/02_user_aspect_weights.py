"""
Step 2: per-user aspect importance + traveler personas (K-means)

(1) Global linear regression
    Fit  rating ≈ w0 + Σ_k w_k · aspect_k  on train reviews that have all 6 sub-ratings.
    w_k is "how much aspect k weighs in the overall rating", i.e. the aspect importance of an average user.

(2) Per-user Ridge regression (shrunk toward the global weights)
    A user has only 3-4 complete reviews on average, so fitting 7 coefficients per user overfits.
    Instead of plain Ridge (pulling coefficients toward 0) we pull them toward the global weights w_g.
        minimize  Σ_i (y_i - x_i·w_u)^2 + λ ||w_u - w_g||^2
        solution: w_u = (X_uᵀX_u + λI)^(-1) (X_uᵀy_u + λ w_g)
    Users with few reviews stay close to w_g; users with many reviews follow their own data.
    Users with no complete review get w_g.
    λ is the value with the lowest RMSE when predicting the rating of val reviews (all sub-ratings)
    from their own aspects.

(3) Personas
    Weights are turned into importance shares (negatives clipped to 0, summing to 1) and clustered with K-means.
    Only users with at least MIN_FULL complete reviews are used for fitting (the rest are almost w_g and
    would distort the clusters). K is chosen by silhouette score; every user is then assigned to the
    nearest cluster.

Output: data/pipeline/user_weights.npy   (n_users, 7) = [w0, service, cleanliness, value, rooms, location, sleep_quality]
        data/pipeline/global_weights.npy (7,) global weights
        data/pipeline/user_personas.csv  user_id, n_full, persona, 6 importance shares
        data/pipeline/persona_centroids.csv
        Pipeline/outputs/  02_*.csv, figures/02_*.png
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

from common import (ASPECTS, SEED, WORK_DIR, ensure_dirs, load_split, rmse, save_fig, save_table,
                    setup_plot)

LAMBDAS = [0.3, 1, 3, 10, 30, 100, 300, 1000]
BUCKETS = [0, 1, 3, 7, 15, np.inf]   # bins of the user's number of complete train reviews (effect of λ per bin)
MIN_FULL = 3          # minimum complete reviews for a user to be used in persona fitting


def design(df):
    """[1, service, ..., sleep_quality] matrix"""
    return np.column_stack([np.ones(len(df)), df[ASPECTS].to_numpy(float)])


def fit_user_weights(X, y, uid, n_users, w_g, lam):
    """Solve all per-user normal equations at once (n_users 7x7 systems)"""
    d = X.shape[1]
    G = np.zeros((n_users, d, d))
    b = np.zeros((n_users, d))
    # np.add.at(array, indices, values): accumulating add that sums repeated indices
    np.add.at(G, uid, X[:, :, None] * X[:, None, :])     # Σ x xᵀ
    np.add.at(b, uid, X * y[:, None])                    # Σ x y
    A = G + lam * np.eye(d)
    rhs = b + lam * w_g
    return np.linalg.solve(A, rhs[..., None])[..., 0]   # users with no data get w_g as the solution


def to_share(W):
    """6 regression coefficients -> importance shares (negatives set to 0, sum to 1)"""
    s = np.clip(W[:, 1:], 0, None)
    tot = s.sum(1, keepdims=True)
    return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1 / len(ASPECTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="number of personas (if omitted, chosen from 3-8 by silhouette score)")
    args = ap.parse_args()
    ensure_dirs()
    plt = setup_plot()

    df, n_users, _ = load_split()
    full = df.dropna(subset=ASPECTS)
    tr, va = full[full.split == "train"], full[full.split == "val"]
    X, y, uid = design(tr), tr.rating.to_numpy(float), tr.user_id.to_numpy()
    print(f"complete-aspect train reviews: {len(tr):,}, users: {tr.user_id.nunique():,}")

    # (1) global linear regression
    lr = LinearRegression().fit(X[:, 1:], y)
    w_g = np.r_[lr.intercept_, lr.coef_]
    print(f"global weights: w0={w_g[0]:.3f}, " + ", ".join(f"{a}={w:.3f}" for a, w in zip(ASPECTS, w_g[1:])))
    print(f"global regression R^2 = {lr.score(X[:, 1:], y):.3f}")

    # (2) choose λ: predict the rating of complete val reviews from their own aspects
    Xv, yv, uv = design(va), va.rating.to_numpy(float), va.user_id.to_numpy()
    n_full = np.bincount(uid, minlength=n_users)
    bucket = pd.cut(n_full[uv], BUCKETS, right=False).astype(str)

    def row(name, pred):
        r = {"lambda": name, "val_rmse": rmse(yv, pred)}
        for b in sorted(set(bucket), key=lambda s: float(s[1:].split(",")[0])):   # RMSE per bin
            m = bucket == b
            r[f"n_full {b} (n={m.sum()})"] = rmse(yv[m], pred[m])
        return r

    rows = [row("global", Xv @ w_g)]
    best = (np.inf, None, None)
    for lam in LAMBDAS:
        W = fit_user_weights(X, y, uid, n_users, w_g, lam)
        pred = (Xv * W[uv]).sum(1)
        rows.append(row(lam, pred))
        if rmse(yv, pred) < best[0]:
            best = (rmse(yv, pred), lam, W)
    lam_tab = pd.DataFrame(rows)
    print(lam_tab.round(4).to_string(index=False))
    _, lam, W = best
    print(f"selected λ = {lam}")
    save_table(lam_tab, "02_lambda_selection")
    np.save(WORK_DIR / "user_weights.npy", W)
    np.save(WORK_DIR / "global_weights.npy", w_g)

    # (3) personas
    share = to_share(W)
    mask = n_full >= MIN_FULL
    S = share[mask]
    print(f"users for persona fitting ({MIN_FULL}+ complete reviews): {mask.sum():,}")

    rng = np.random.default_rng(SEED)
    sample = S[rng.choice(len(S), size=min(20_000, len(S)), replace=False)]
    sil = []
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
        sil.append({"k": k, "silhouette": silhouette_score(sample, km.predict(sample)), "inertia": km.inertia_})
    sil = pd.DataFrame(sil)
    print(sil.to_string(index=False))
    save_table(sil, "02_kmeans_k_selection")
    k = args.k or int(sil.loc[sil.silhouette.idxmax(), "k"])
    print(f"selected K = {k}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(sil.k, sil.inertia, marker="o")
    ax[0].set_title("K-means inertia (elbow)")
    ax[1].plot(sil.k, sil.silhouette, marker="o", color="C1")
    ax[1].axvline(k, ls="--", color="gray")
    ax[1].set_title("Silhouette score")
    for a in ax:
        a.set_xlabel("K")
    save_fig(plt, "02_kmeans_k_selection")

    # KMeans(n_clusters, n_init=best of several initializations, random_state=reproducibility)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
    persona = km.predict(share)                 # assign every user to the nearest cluster
    cent = pd.DataFrame(km.cluster_centers_, columns=ASPECTS)
    base = S.mean(0)
    # name: the aspect with the highest share relative to the overall average
    cent["name"] = [f"P{i}_{ASPECTS[np.argmax(c / base)]}" for i, c in enumerate(km.cluster_centers_)]
    cent["n_users_fit"] = np.bincount(km.labels_, minlength=k)
    cent["n_users_all"] = np.bincount(persona, minlength=k)
    cent.insert(0, "persona", range(k))
    print(cent.round(3).to_string(index=False))
    cent.to_csv(WORK_DIR / "persona_centroids.csv", index=False)
    save_table(cent.round(4), "02_persona_centroids")

    up = pd.DataFrame(share, columns=ASPECTS)
    up.insert(0, "persona", persona)
    up.insert(0, "n_full", n_full)
    up.insert(0, "user_id", np.arange(n_users))
    up.to_csv(WORK_DIR / "user_personas.csv", index=False)

    # figure: global weights, importance shares per persona
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(ASPECTS, w_g[1:], color="#4c72b0")
    ax[0].set_title("Global aspect weights (linear regression coefficients)")
    ax[0].tick_params(axis="x", rotation=30)
    dev = cent[ASPECTS].to_numpy() - base
    im = ax[1].imshow(dev, cmap="RdBu_r", vmin=-np.abs(dev).max(), vmax=np.abs(dev).max())
    ax[1].set_xticks(range(len(ASPECTS)), ASPECTS, rotation=30)
    ax[1].set_yticks(range(k), cent["name"] + " (" + cent.n_users_fit.astype(str) + ")")
    for i in range(k):
        for j in range(len(ASPECTS)):
            ax[1].text(j, i, f"{cent[ASPECTS].iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[1].set_title("Importance share per persona (color: difference from overall average)")
    plt.colorbar(im, ax=ax[1])
    save_fig(plt, "02_aspect_weights_personas")


if __name__ == "__main__":
    main()
