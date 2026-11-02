"""
4단계: 편차 기반 사용자 선호와 여행자 페르소나 재구성

배경
    02_user_aspect_weights.py는 사용자별 회귀 계수(전역 가중치 쪽으로 축소한 Ridge)로 aspect 중요도를 만들었다.
    그러나 사용자당 세부 평점 완전 리뷰가 평균 3.7건뿐이고 6개 aspect끼리 상관이 높아,
    선택된 λ=1000에서 개인 가중치가 전역 가중치와 거의 같아졌고 페르소나 3개의 중요도 비율 차이도 0.003 수준이었다.

아이디어 (팀 논의 2번의 선택지 (c))
    회귀 계수(7개를 추정해야 함) 대신, 사용자가 각 aspect에 준 점수가 "그 호텔의 평균 aspect 점수"와
    얼마나 다른지를 직접 평균한다. 계수 추정이 아니라 평균이므로 리뷰가 3~4건이어도 안정적이다.

    (1) 원 편차   d_uk  = mean_i [ score_{u,i,k} - hotel_mean_{h(i),k} ]       (train 리뷰, aspect k가 있는 리뷰만)
    (2) 관대함    L_u   = 사용자 u의 모든 aspect 편차의 평균                     (모든 항목을 후하게/박하게 주는 성향)
    (3) 상대 편차 c_uk  = d_uk - L_u                                             (관대함을 뺀, aspect별 상대적 까다로움)
    (4) 축소      c_uk  ← c_uk · n_uk / (n_uk + M)                               (리뷰가 적은 aspect는 0 쪽으로)

    c_uk < 0 : 다른 사람보다 그 aspect를 박하게 평가 = 까다롭게 보는 aspect (중시하는 aspect로 해석)
    c_uk > 0 : 그 aspect에는 관대함 = 덜 신경 쓰는 aspect

    "까다로움(strictness)" = -c_uk 로 바꿔 K-means로 군집화하면 "청결 깐깐형", "가성비 깐깐형" 같은 페르소나가 된다.

검증 (이 표현이 실제로 사용자의 안정적인 성향인지)
    val 리뷰의 aspect 점수를 예측해 RMSE를 비교한다 (val은 사용자의 train 이후 리뷰라 시간 누수가 없다).
        P0  호텔 평균                                   hotel_mean_hk
        P1  + 관대함                                     hotel_mean_hk + L_u
        P2  + 관대함 + 페르소나 중심 편차                hotel_mean_hk + L_u + centroid_{p(u),k}
        P3  + 관대함 + 개인 상대 편차                    hotel_mean_hk + L_u + c_uk
    P3 < P1 이면 관대함 말고도 aspect별 개인 성향이 존재한다는 뜻이고,
    P2가 P1보다 좋아지면 페르소나가 그 성향을 일부라도 담고 있다는 뜻이다.

입력: data/pipeline/ratings_split.csv   (01_split.py)
      data/pipeline/hotel_aspects.npy   (03_hotel_features.py, train 리뷰만으로 만든 호텔 aspect 베이지안 평균)
출력: data/pipeline/user_personas_dev.csv, persona_dev_centroids.csv
      Pipeline/outputs/04_*.csv, figures/04_*.png

실행:
      cd Pipeline && python 04_persona_deviation.py          (K 자동 선택)
      cd Pipeline && python 04_persona_deviation.py --k 4    (K 지정)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import ASPECTS, SEED, WORK_DIR, ensure_dirs, load_split, rmse, save_fig, save_table, setup_plot

M = 3                 # 축소 강도: aspect별 리뷰가 M건이면 편차를 절반으로 줄인다 (가상 리뷰 수)
MIN_PER_ASPECT = 3    # 페르소나 학습에 쓰는 사용자: 6개 aspect 모두 train 리뷰가 이 개수 이상
K_RANGE = range(2, 9)


def user_deviations(tr, A, n_users):
    """
    사용자별 aspect 편차 통계
    반환: dev_sum (n_users, 6), cnt (n_users, 6)
          dev_sum[u, k] = Σ (사용자 점수 - 호텔 평균),  cnt[u, k] = 그 aspect가 있는 리뷰 수
    """
    K = len(ASPECTS)
    dev_sum = np.zeros((n_users, K))
    cnt = np.zeros((n_users, K))
    for k, a in enumerate(ASPECTS):
        s = tr[["user_id", "hotel_id", a]].dropna()
        u = s.user_id.to_numpy()
        r = s[a].to_numpy(float) - A[s.hotel_id.to_numpy(), k]
        # np.bincount(인덱스, weights=값, minlength=길이): 같은 인덱스의 값을 모두 더한 배열
        dev_sum[:, k] = np.bincount(u, weights=r, minlength=n_users)
        cnt[:, k] = np.bincount(u, minlength=n_users)
    return dev_sum, cnt


def build_profiles(dev_sum, cnt):
    """
    관대함 L_u(축소)와 상대 편차 c_uk(축소)를 만든다.
    관대함은 모든 aspect 편차를 합쳐 계산하고 전체 리뷰 수로 축소한다.
    """
    n_all = cnt.sum(1)
    L_raw = dev_sum.sum(1) / np.maximum(n_all, 1)
    L = L_raw * n_all / (n_all + M)                              # 리뷰가 없으면 0 (= 평균적인 사용자)
    d = np.where(cnt > 0, dev_sum / np.maximum(cnt, 1), 0.0)      # aspect별 원 편차
    c = np.where(cnt > 0, d - L_raw[:, None], 0.0)                # 관대함을 뺀 상대 편차
    c = c * cnt / (cnt + M)
    return L, c


def val_rmse_table(va, A, L, c, persona, cent_c):
    """val 리뷰의 aspect 점수 예측 RMSE (aspect별 + 전체)"""
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
    """프로필 계산 → K 선택·군집화 → 페르소나 저장 → val 검증 → 그림 저장"""
    # argparse.ArgumentParser: 명령줄 인자 해석기, add_argument(이름, type=형, default=기본값, help=설명)
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="페르소나 수 (지정하지 않으면 실루엣 점수로 선택)")
    args = ap.parse_args()
    ensure_dirs()
    plt = setup_plot()

    df, n_users, n_hotels = load_split()
    A = np.load(WORK_DIR / "hotel_aspects.npy")
    assert A.shape == (n_hotels, len(ASPECTS)), f"hotel_aspects.npy 크기 {A.shape}가 호텔 수 {n_hotels}와 다릅니다"
    tr, va = df[df.split == "train"], df[df.split == "val"]

    # (1)~(4) 사용자 프로필
    dev_sum, cnt = user_deviations(tr, A, n_users)
    L, c = build_profiles(dev_sum, cnt)
    strict = -c                                                     # 양수 = 그 aspect를 까다롭게 봄
    print(f"관대함 L_u: 평균 {L.mean():+.3f}, 표준편차 {L.std():.3f}")
    print("상대 편차 표준편차 (aspect별):", dict(zip(ASPECTS, c.std(0).round(3))))

    # 페르소나 학습 사용자
    mask = (cnt >= MIN_PER_ASPECT).all(1)
    S = strict[mask]
    print(f"페르소나 학습 사용자 (6개 aspect 모두 {MIN_PER_ASPECT}건 이상): {mask.sum():,}명 / 전체 {n_users:,}명")

    # K 선택: 실루엣 점수 + 중심 간 차이(기존 방식은 0.003 수준이었음)
    rng = np.random.default_rng(SEED)
    sample = S[rng.choice(len(S), size=min(20_000, len(S)), replace=False)]
    rows = []
    for k in K_RANGE:
        # KMeans(n_clusters=군집 수, n_init=초기값 시도 횟수 중 최선 선택, random_state=재현성)
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
        spread = np.abs(km.cluster_centers_ - S.mean(0)).max()      # 중심이 전체 평균에서 가장 멀리 떨어진 정도
        # silhouette_score(데이터, 군집 라벨): -1~1, 클수록 군집이 잘 분리됨 (계산량 때문에 2만 명 표본 사용)
        rows.append({"k": k, "silhouette": silhouette_score(sample, km.predict(sample)),
                     "inertia": km.inertia_, "max_centroid_dev": spread})
    ksel = pd.DataFrame(rows)
    print(ksel.round(4).to_string(index=False))
    save_table(ksel, "04_kmeans_k_selection")
    k = args.k or int(ksel.loc[ksel.silhouette.idxmax(), "k"])
    print(f"선택 K = {k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
    persona = km.predict(strict)                                    # 모든 사용자에게 가장 가까운 군집 배정
    cent = pd.DataFrame(km.cluster_centers_, columns=ASPECTS)
    # 이름: 가장 까다롭게 보는 aspect (중심의 strictness가 가장 큰 aspect)
    cent["name"] = [f"P{i}_strict_{ASPECTS[int(np.argmax(r))]}" for i, r in enumerate(km.cluster_centers_)]
    cent["n_users_fit"] = np.bincount(km.labels_, minlength=k)
    cent["n_users_all"] = np.bincount(persona, minlength=k)
    cent.insert(0, "persona", range(k))
    print(cent.round(3).to_string(index=False))
    cent.to_csv(WORK_DIR / "persona_dev_centroids.csv", index=False)
    save_table(cent.round(4), "04_persona_dev_centroids")

    # 중요도 비율(02와 같은 형식, 합 1) 버전도 저장: 08_kaggle_generalization.py 등에 그대로 꽂아 쓰기 위함
    # 전역 가중치 비율에 까다로움을 반영해 기울인다: share_k ∝ w_g,k · exp(strictness_k)
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

    # 검증: val aspect 점수 예측
    cent_c = -km.cluster_centers_                                   # 중심을 상대 편차(c) 단위로 되돌림
    tab = val_rmse_table(va, A, L, c, persona, cent_c)
    print(tab.round(4).to_string(index=False))
    save_table(tab.round(5), "04_val_aspect_rmse")

    # 그림 1: K 선택
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ksel.k, ksel.silhouette, marker="o", color="C1")
    ax[0].axvline(k, ls="--", color="gray")
    ax[0].set_title("실루엣 점수")
    ax[1].plot(ksel.k, ksel.max_centroid_dev, marker="o")
    ax[1].set_title("중심과 전체 평균의 최대 차이 (점수 단위)")
    for a in ax:
        a.set_xlabel("K")
    save_fig(plt, "04_kmeans_k_selection")

    # 그림 2: 페르소나별 까다로움
    C = cent[ASPECTS].to_numpy()
    lim = max(np.abs(C).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 0.6 * k + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(ASPECTS)), ASPECTS, rotation=30)
    ax.set_yticks(range(k), cent["name"] + " (" + cent.n_users_fit.astype(str) + ")")
    for i in range(k):
        for j in range(len(ASPECTS)):
            ax.text(j, i, f"{C[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("페르소나별 aspect 까다로움 (+: 호텔 평균보다 박하게 평가)")
    plt.colorbar(im)
    save_fig(plt, "04_persona_dev_strictness")


if __name__ == "__main__":
    main()
