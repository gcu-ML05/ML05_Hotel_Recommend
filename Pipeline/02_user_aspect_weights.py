"""
2단계: 사용자별 aspect 중요도 추정 + 여행자 페르소나(K-means)

(1) 전역 선형회귀
    세부 평점이 6개 모두 있는 train 리뷰로  rating ≈ w0 + Σ_k w_k · aspect_k  를 적합한다.
    w_k가 "전체 평점을 정할 때 aspect k가 차지하는 비중", 즉 평균적인 사용자의 aspect 중요도이다.

(2) 사용자별 Ridge 회귀 (전역 가중치 쪽으로 축소)
    사용자 u의 완전 리뷰가 평균 3~4건뿐이라 7개 계수를 사용자마다 따로 적합하면 과적합된다.
    그래서 일반 Ridge(계수를 0 쪽으로 당김) 대신 계수를 전역 가중치 w_g 쪽으로 당기는 정규화를 쓴다.
        minimize  Σ_i (y_i - x_i·w_u)^2 + λ ||w_u - w_g||^2
        해:       w_u = (X_uᵀX_u + λI)^(-1) (X_uᵀy_u + λ w_g)
    리뷰가 적은 사용자는 w_g에 가깝고, 많을수록 자기 데이터에 맞춰진다. 완전 리뷰가 없는 사용자는 w_g를 쓴다.
    λ는 val 리뷰(세부 평점 완전)의 rating을 그 리뷰의 aspect로 예측한 RMSE가 가장 작은 값으로 고른다.

(3) 페르소나
    가중치를 중요도 비율(음수는 0으로 자르고 합이 1이 되게)로 바꿔 K-means로 군집화한다.
    완전 리뷰가 MIN_FULL건 이상인 사용자로만 학습한다 (나머지는 거의 w_g 그대로라 군집을 왜곡하므로).
    K는 실루엣 점수로 고르고, 학습 후 모든 사용자에게 가장 가까운 군집을 배정한다.

출력: data/pipeline/user_weights.npy   (n_users, 7) = [w0, service, cleanliness, value, rooms, location, sleep_quality]
      data/pipeline/global_weights.npy (7,) 전역 가중치
      data/pipeline/user_personas.csv  user_id, n_full, persona, 중요도 비율 6개
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
BUCKETS = [0, 1, 3, 7, 15, np.inf]   # 사용자의 train 완전 리뷰 수 구간 (λ 효과를 구간별로 확인)
MIN_FULL = 3          # 페르소나 학습에 쓰는 사용자의 최소 완전 리뷰 수


def design(df):
    """[1, service, ..., sleep_quality] 행렬"""
    return np.column_stack([np.ones(len(df)), df[ASPECTS].to_numpy(float)])


def fit_user_weights(X, y, uid, n_users, w_g, lam):
    """사용자별 정규방정식을 한 번에 푼다 (7x7 행렬 n_users개)"""
    d = X.shape[1]
    G = np.zeros((n_users, d, d))
    b = np.zeros((n_users, d))
    # np.add.at(배열, 인덱스, 값): 같은 인덱스가 여러 번 나와도 모두 더해 주는 누적 덧셈
    np.add.at(G, uid, X[:, :, None] * X[:, None, :])     # Σ x xᵀ
    np.add.at(b, uid, X * y[:, None])                    # Σ x y
    A = G + lam * np.eye(d)
    rhs = b + lam * w_g
    return np.linalg.solve(A, rhs[..., None])[..., 0]   # 데이터가 없는 사용자는 해가 w_g


def to_share(W):
    """회귀 계수 6개 -> 중요도 비율 (음수 0 처리, 합 1)"""
    s = np.clip(W[:, 1:], 0, None)
    tot = s.sum(1, keepdims=True)
    return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1 / len(ASPECTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="페르소나 수 (지정하지 않으면 실루엣 점수로 3~8 중 선택)")
    args = ap.parse_args()
    ensure_dirs()
    plt = setup_plot()

    df, n_users, _ = load_split()
    full = df.dropna(subset=ASPECTS)
    tr, va = full[full.split == "train"], full[full.split == "val"]
    X, y, uid = design(tr), tr.rating.to_numpy(float), tr.user_id.to_numpy()
    print(f"세부 평점 완전 train 리뷰 {len(tr):,}건, 사용자 {tr.user_id.nunique():,}명")

    # (1) 전역 선형회귀
    lr = LinearRegression().fit(X[:, 1:], y)
    w_g = np.r_[lr.intercept_, lr.coef_]
    print(f"전역 가중치: w0={w_g[0]:.3f}, " + ", ".join(f"{a}={w:.3f}" for a, w in zip(ASPECTS, w_g[1:])))
    print(f"전역 회귀 R^2 = {lr.score(X[:, 1:], y):.3f}")

    # (2) λ 선택: val 완전 리뷰의 rating을 자기 aspect로 예측
    Xv, yv, uv = design(va), va.rating.to_numpy(float), va.user_id.to_numpy()
    n_full = np.bincount(uid, minlength=n_users)
    bucket = pd.cut(n_full[uv], BUCKETS, right=False).astype(str)

    def row(name, pred):
        r = {"lambda": name, "val_rmse": rmse(yv, pred)}
        for b in sorted(set(bucket), key=lambda s: float(s[1:].split(",")[0])):   # 구간별 RMSE
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
    print(f"선택 λ = {lam}")
    save_table(lam_tab, "02_lambda_selection")
    np.save(WORK_DIR / "user_weights.npy", W)
    np.save(WORK_DIR / "global_weights.npy", w_g)

    # (3) 페르소나
    share = to_share(W)
    mask = n_full >= MIN_FULL
    S = share[mask]
    print(f"페르소나 학습 사용자 (완전 리뷰 {MIN_FULL}건 이상): {mask.sum():,}명")

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
    print(f"선택 K = {k}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(sil.k, sil.inertia, marker="o")
    ax[0].set_title("K-means inertia (elbow)")
    ax[1].plot(sil.k, sil.silhouette, marker="o", color="C1")
    ax[1].axvline(k, ls="--", color="gray")
    ax[1].set_title("실루엣 점수")
    for a in ax:
        a.set_xlabel("K")
    save_fig(plt, "02_kmeans_k_selection")

    # KMeans(n_clusters, n_init=여러 초기값 중 최선 선택, random_state=재현성)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
    persona = km.predict(share)                 # 모든 사용자에게 가장 가까운 군집 배정
    cent = pd.DataFrame(km.cluster_centers_, columns=ASPECTS)
    base = S.mean(0)
    # 이름: 전체 평균 대비 비율이 가장 높은 aspect
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

    # 그림: 전역 가중치, 페르소나별 중요도 비율
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(ASPECTS, w_g[1:], color="#4c72b0")
    ax[0].set_title("전역 aspect 가중치 (선형회귀 계수)")
    ax[0].tick_params(axis="x", rotation=30)
    dev = cent[ASPECTS].to_numpy() - base
    im = ax[1].imshow(dev, cmap="RdBu_r", vmin=-np.abs(dev).max(), vmax=np.abs(dev).max())
    ax[1].set_xticks(range(len(ASPECTS)), ASPECTS, rotation=30)
    ax[1].set_yticks(range(k), cent["name"] + " (" + cent.n_users_fit.astype(str) + ")")
    for i in range(k):
        for j in range(len(ASPECTS)):
            ax[1].text(j, i, f"{cent[ASPECTS].iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[1].set_title("페르소나별 중요도 비율 (색: 전체 평균 대비 차이)")
    plt.colorbar(im, ax=ax[1])
    save_fig(plt, "02_aspect_weights_personas")


if __name__ == "__main__":
    main()
