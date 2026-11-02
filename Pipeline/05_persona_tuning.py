"""
5단계: 편차 기반 페르소나(04)의 개선 실험과 최종 test 평가

04에서 남은 문제
    (1) 축소 강도 M=3을 근거 없이 정했다. P3(개인 편차)가 P1(관대함만)보다 나빴는데, 축소가 약해 잡음이 섞였을 수 있다.
    (2) 관대함 L_u를 모든 aspect에 같은 크기로 더했다. location은 P1이 P0(호텔 평균)보다 나빠서
        aspect마다 관대함이 반영되는 정도가 다르다는 신호가 있었다.
    (3) K를 실루엣으로만 골랐다 (K=2). 해석 가능성과 예측력으로 K=3, 4도 비교해야 한다.
    (4) 하이퍼파라미터를 val로 고르면서 val로 보고했다. 최종 수치는 test로 따로 보고해야 한다.

이 스크립트가 하는 일
    A. M 선택 (val):        c_uk ← c_uk · n_uk / (n_uk + M),  M ∈ M_GRID 중 val P3 RMSE가 가장 낮은 값
    B. aspect별 관대함 계수 (val로 추정):
           β_k = argmin_β Σ (y - hotel_mean_k - β · L_u)²  =  Σ L_u (y - hotel_mean_k) / Σ L_u²    (최소제곱 닫힌 해)
           P1β = hotel_mean_k + β_k · L_u
    C. K 비교 (val):         K ∈ K_GRID, 실루엣·중심 이탈·P2 RMSE·가장 작은 군집 비율
    D. 최종 test 평가:       A~C에서 고른 설정을 고정하고 test 리뷰로 P0~P3 RMSE 보고 (프로필은 train만 사용)
    E. 페르소나 요약 표:     사람이 읽는 이름, 인원, 평균 관대함, 평균 평점, 평균 리뷰 수

입력: data/pipeline/ratings_split.csv, hotel_aspects.npy (04와 같음)
출력: Pipeline/outputs/05_*.csv, figures/05_*.png
      data/pipeline/user_personas_dev_tuned.csv (06 학습·평가에서 쓸 수 있는 형식, 선택된 M·K 적용)
실행: cd Pipeline && python 05_persona_tuning.py
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import ASPECTS, SEED, WORK_DIR, ensure_dirs, load_split, rmse, save_fig, save_table, setup_plot

# 파일명이 숫자로 시작해 import 문을 쓸 수 없으므로 importlib.import_module(모듈명)으로 04의 함수를 재사용
dev04 = importlib.import_module("04_persona_deviation")

M_L = dev04.M                         # 관대함 축소 강도는 04와 같게 고정 (L은 6개 aspect를 합쳐 건수가 많아 영향이 작음)
M_GRID = [1, 3, 10, 30, 100, 300]     # 상대 편차 축소 강도 후보 (M=0은 리뷰 0건 aspect에서 0/0이라 제외)
K_GRID = [2, 3, 4, 5]
K_TOL = 0.0005                        # val RMSE가 최솟값과 이 차이 이내면 더 작은(해석하기 쉬운) K를 고른다
KR = {"service": "서비스", "cleanliness": "청결", "value": "가성비",
      "rooms": "객실", "location": "위치", "sleep_quality": "수면"}


def profiles(dev_sum, cnt, m_c):
    """
    관대함 L_u와 상대 편차 c_uk. 04의 build_profiles와 같고, c의 축소 강도만 m_c로 바꿀 수 있다.
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
    """aspect별 (사용자, 호텔 평균, 실제 점수) 배열. 결측 aspect는 제외."""
    out = {}
    for k, a in enumerate(ASPECTS):
        s = df[["user_id", "hotel_id", a]].dropna()
        out[a] = (s.user_id.to_numpy(), A[s.hotel_id.to_numpy(), k], s[a].to_numpy(float))
    return out


def rmse_table(arr, preds):
    """
    preds: {이름: f(k, u, base) -> 예측}. aspect별 RMSE와 전체(ALL, 모든 aspect 점수를 합쳐 계산) RMSE 표.
    예측은 평점 범위 [1, 5]로 자른다 (np.clip(값, 최솟값, 최댓값)).
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
    """aspect별 관대함 계수 β_k = Σ L_u (y - base) / Σ L_u²  (절편 없는 단순 회귀의 최소제곱 해)"""
    beta = np.ones(len(ASPECTS))
    for k, a in enumerate(ASPECTS):
        u, base, y = arr[a]
        beta[k] = (L[u] * (y - base)).sum() / max((L[u] ** 2).sum(), 1e-12)
    return beta


def persona_name(center):
    """중심 벡터(까다로움) → 사람이 읽는 이름. 가장 까다로운 aspect + 가장 관대한 aspect."""
    hi, lo = ASPECTS[int(np.argmax(center))], ASPECTS[int(np.argmin(center))]
    return f"{KR[hi]} 깐깐형 ({KR[lo]}엔 관대)"


def fit_personas(strict, mask, k):
    """6개 aspect 모두 리뷰가 충분한 사용자로 K-means 학습 후 전원 배정. 반환: (모델, 전체 배정)"""
    # KMeans(n_clusters=군집 수, n_init=초기값 시도 횟수 중 최선 선택, random_state=재현성)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(strict[mask])
    return km, km.predict(strict)


def main():
    """A~E 실험을 순서대로 실행하고 표·그림을 저장한다."""
    ensure_dirs()
    plt = setup_plot()
    df, n_users, n_hotels = load_split()
    A = np.load(WORK_DIR / "hotel_aspects.npy")
    tr = df[df.split == "train"]
    va_arr = aspect_arrays(df[df.split == "val"], A)
    te_arr = aspect_arrays(df[df.split == "test"], A)

    dev_sum, cnt = dev04.user_deviations(tr, A, n_users)
    mask = (cnt >= dev04.MIN_PER_ASPECT).all(1)

    # ---------------- A. 축소 강도 M 선택 (val) ----------------
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
    print(f"선택 M = {m_best}")
    L, c = profiles(dev_sum, cnt, m_best)
    strict = -c

    # ---------------- B. aspect별 관대함 계수 β (val) ----------------
    beta = fit_beta(va_arr, L)
    btab = pd.DataFrame({"aspect": ASPECTS, "beta": beta})
    print(btab.round(3).to_string(index=False))
    save_table(btab.round(4), "05_leniency_beta")

    # ---------------- C. K 비교 (val) ----------------
    rng = np.random.default_rng(SEED)
    idx = rng.choice(mask.sum(), size=min(20_000, mask.sum()), replace=False)
    rows, models = [], {}
    for k in K_GRID:
        km, persona = fit_personas(strict, mask, k)
        models[k] = (km, persona)
        cc = -km.cluster_centers_                                   # 상대 편차(c) 단위 중심
        t = rmse_table(va_arr, {"P2": lambda kk, u, b: b + beta[kk] * L[u] + cc[persona[u], kk]}).set_index("aspect")
        S = strict[mask]
        # silhouette_score(데이터, 군집 라벨): -1~1, 클수록 군집이 잘 분리됨 (2만 명 표본으로 계산)
        rows.append({"k": k, "silhouette": silhouette_score(S[idx], km.predict(S[idx])),
                     "max_centroid_dev": np.abs(km.cluster_centers_ - S.mean(0)).max(),
                     "smallest_share": np.bincount(persona, minlength=k).min() / n_users,
                     "val_P2": t.loc["ALL", "P2"], "val_P2_location": t.loc["location", "P2"],
                     "names": " / ".join(persona_name(r) for r in km.cluster_centers_)})
    ktab = pd.DataFrame(rows)
    print(ktab.round(4).to_string(index=False))
    save_table(ktab.round(5), "05_k_comparison")
    # 간결성 규칙: val P2가 최솟값 + K_TOL 이내인 K 중 가장 작은 값 (차이가 미미한데 큰 K를 고르면 val에 과적합)
    k_best = int(ktab.loc[ktab.val_P2 <= ktab.val_P2.min() + K_TOL, "k"].min())
    print(f"선택 K = {k_best} (val P2 RMSE 최솟값 + {K_TOL} 이내 중 가장 작은 K)")
    km, persona = models[k_best]
    cc = -km.cluster_centers_

    # 10(Kaggle 비교)에 꽂을 비율 형식 (04와 같은 변환: share_k ∝ 전역 가중치_k · exp(까다로움_k))
    w_g = np.clip(np.load(WORK_DIR / "global_weights.npy")[1:], 1e-6, None)
    tilt = w_g * np.exp(km.cluster_centers_)
    share = pd.DataFrame(tilt / tilt.sum(1, keepdims=True), columns=ASPECTS)
    share.insert(0, "persona", range(k_best))
    share["name"] = [f"T{i}_{persona_name(r)}" for i, r in enumerate(km.cluster_centers_)]
    share["n_users_fit"] = np.bincount(km.labels_, minlength=k_best)
    save_table(share.round(4), "05_persona_dev_share")

    # ---------------- D. 최종 test 평가 ----------------
    L0, c0 = profiles(dev_sum, cnt, M_L)                             # 04 원래 설정 (M=3, β=1, K=2)
    km0, persona0 = fit_personas(-c0, (cnt >= dev04.MIN_PER_ASPECT).all(1), 2)
    cc0 = -km0.cluster_centers_
    preds = {
        "P0 호텔 평균": lambda k, u, b: b,
        "P1 +관대함 (04)": lambda k, u, b: b + L0[u],
        "P2 +페르소나 (04: M=3, K=2)": lambda k, u, b: b + L0[u] + cc0[persona0[u], k],
        "P3 +개인 편차 (04: M=3)": lambda k, u, b: b + L0[u] + c0[u, k],
        "P1β aspect별 관대함": lambda k, u, b: b + beta[k] * L[u],
        f"P2β +페르소나 (M={m_best}, K={k_best})": lambda k, u, b: b + beta[k] * L[u] + cc[persona[u], k],
        f"P3β +개인 편차 (M={m_best})": lambda k, u, b: b + beta[k] * L[u] + c[u, k],
    }
    test_tab = rmse_table(te_arr, preds)
    val_tab = rmse_table(va_arr, preds)
    print("test\n", test_tab.round(4).to_string(index=False))
    save_table(test_tab.round(5), "05_test_aspect_rmse")
    save_table(val_tab.round(5), "05_val_aspect_rmse")

    # ---------------- E. 페르소나 요약 표 ----------------
    rating_sum = np.bincount(tr.user_id, weights=tr.rating, minlength=n_users)
    rating_cnt = np.bincount(tr.user_id, minlength=n_users)
    summ = []
    for label, (kmx, pers, Lx) in {"04 원래 (M=3, K=2)": (km0, persona0, L0),
                                   f"개선 (M={m_best}, K={k_best})": (km, persona, L)}.items():
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

    # 다른 단계에서 쓸 사용자별 페르소나 파일 (개선 설정)
    users = pd.DataFrame(strict, columns=[f"strict_{a}" for a in ASPECTS])
    users.insert(0, "leniency", L)
    users.insert(0, "persona_name", [persona_name(r) for r in km.cluster_centers_[persona]])
    users.insert(0, "persona", persona)
    users.insert(0, "user_id", np.arange(n_users))
    users.to_csv(WORK_DIR / "user_personas_dev_tuned.csv", index=False)

    # 그림: M에 따른 val RMSE
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = np.arange(len(M_GRID))
    ax.plot(xs, mtab.val_P1, marker="o", label="P1 +관대함")
    ax.plot(xs, mtab.val_P3, marker="o", label="P3 +관대함+개인 편차")
    ax.set_xticks(xs, [str(m) for m in M_GRID])
    ax.set_xlabel("축소 강도 M")
    ax.set_ylabel("val RMSE (전체 aspect)")
    ax.set_title("상대 편차 축소 강도에 따른 val RMSE")
    ax.legend()
    save_fig(plt, "05_shrinkage_M")

    # 그림: 선택된 K의 페르소나 까다로움
    C = km.cluster_centers_
    lim = max(np.abs(C).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 0.6 * k_best + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(ASPECTS)), [KR[a] for a in ASPECTS])
    n = np.bincount(persona, minlength=k_best)
    ax.set_yticks(range(k_best), [f"{persona_name(r)} ({n[i]:,}명)" for i, r in enumerate(C)])
    for i in range(k_best):
        for j in range(len(ASPECTS)):
            ax.text(j, i, f"{C[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"개선 페르소나 (M={m_best}, K={k_best}) aspect 까다로움")
    plt.colorbar(im)
    save_fig(plt, "05_persona_strictness")


if __name__ == "__main__":
    main()
