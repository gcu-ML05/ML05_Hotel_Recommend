"""
6단계: 모델 학습과 평가 (HotelRec)

학습: train 분할로 모든 모델을 학습한다.
선택: val 분할로 MF 조기 종료, 콘텐츠 기반의 γ, 하이브리드 가중치 λ를 고른다.
평가: test 분할 (사용자별 가장 최근 리뷰)

(1) 평점 예측: 모든 사용자의 val/test 리뷰에 대해 RMSE, MAE
(2) Top-N 추천: 무작위로 뽑은 사용자마다 정답 호텔(test 리뷰의 호텔) 1개 + 그 사용자가 평가한 적 없는
    호텔 N_NEG개를 후보로 두고 점수 순으로 정렬했을 때 HR@k, NDCG@k
    (전체 4.7만 개 호텔을 모두 정렬하는 대신 표본 후보를 쓰는 방식. 후보가 무작위라 절대값보다
     모델 간 비교에 의미가 있다.)
(3) 사용자 활동량(train 리뷰 수)별 HR@10: 리뷰가 적은 사용자에서 콘텐츠 기반/하이브리드가 도움이 되는지
    (제안서 1절의 동기)
(4) 같은 도시 후보 Top-N: 후보를 정답 호텔과 같은 도시의 호텔로만 뽑아 "여행지 안에서 호텔 고르기"를 평가
    (무작위 후보는 인기 호텔과 다른 지역의 비인기 호텔을 구별하기만 하면 되어 인기도 기준선이 지나치게 유리함)
    hotel_map.csv가 리뷰 데이터와 같은 ID 체계일 때만 실행한다.

Item-kNN은 개선판(KNNBaseline + 분모 축소 + 공동 방문 순위 점수)과 기존판(adjusted cosine)을 함께 보고한다.

하이브리드 (제안서 ④):  score = λ · MF + (1 - λ) · ContentBased,  λ는 val NDCG@10이 가장 높은 값

편차 기반 선호 연결 (04·05단계의 user_personas_dev_tuned.csv)
    PersonaPopularity(편차): 05의 페르소나(까다로움 유형)가 같은 사용자들이 4점 이상 준 호텔을 추천
    Content(까다로움): 콘텐츠 기반의 aspect 가중치를 사용자가 까다롭게 보는 aspect 쪽으로 기울인다.
        w_uk = w_g,k · exp(τ · strict_uk),  Σ_k w_uk = Σ_k w_g,k 가 되도록 정규화 (평점 척도 유지)
        절편 = w_g,0 + κ · 관대함_u   (관대함은 사용자마다 상수라 순위는 바꾸지 않고 평점만 보정)
        τ는 val NDCG@10, κ는 val 평점 잔차의 최소제곱으로 고른다.
        까다로움은 세 가지로 비교: 개인 / 페르소나 평균 / 이력이 적은 사용자(train 리뷰 COLD_MAX건 미만)만 페르소나
        (마지막이 역할표의 "콜드스타트 시 페르소나 가중치로 대체")
    Hybrid(까다로움): λ · MF + (1 - λ) · Content(까다로움, 이력 적으면 페르소나), λ는 val NDCG@10

출력: Pipeline/outputs/06_*.csv, figures/06_*.png
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import (ASPECTS, HOTELREC_DIR, SEED, WORK_DIR, ensure_dirs, load_split, mae, ranking_metrics, rmse,
                    sample_candidates, sample_city_candidates, save_fig, save_table, setup_plot)
from models import BiasBaseline, BiasedMF, ContentBased, ItemKNN, PersonaPopularity, Popularity

LAMBDAS = np.round(np.arange(0, 1.01, 0.1), 1)
ACTIVITY_BINS = [3, 5, 10, 20, np.inf]
# 까다로움 반영 강도 후보. 까다로움은 점수 단위로 대부분 ±0.1 이내라 τ가 수십~수백이어야 가중치가 달라진다.
# (가장 까다로운 aspect의 가중치 비율 중앙값: τ=80 → 0.66, 160 → 0.89, 320 → 0.99)
TAUS = [0, 5, 10, 20, 40, 80, 160, 320]
COLD_MAX = 5                        # train 리뷰가 이보다 적은 사용자는 개인 대신 페르소나 까다로움 사용
ACTIVITY_PLOT = ["Popularity", "ItemKNN", "MF", "Content(aspect+text)", "Content(까다로움, 이력 적으면 페르소나)"]


def arrays(df):
    return df.user_id.to_numpy(), df.hotel_id.to_numpy(), df.rating.to_numpy(float)


def strict_weights(w_g, S, tau, leniency=None, kappa=0.0):
    """
    까다로움을 반영한 사용자별 [절편, aspect 가중치 6개] (ContentBased의 W와 같은 형식)
        w_uk ∝ w_g,k · exp(τ · S_uk), 합은 전역 가중치와 같게 맞춤 / 절편 = w_g,0 + κ · 관대함
    """
    w = w_g[1:] * np.exp(tau * S)
    w *= w_g[1:].sum() / w.sum(1, keepdims=True)
    w0 = w_g[0] + (kappa * leniency if leniency is not None else 0.0)
    return np.column_stack([np.broadcast_to(w0, len(S)), w])


def cache_pairs(f):
    """같은 (사용자, 호텔) 배열에 대한 결과를 저장해 두고 재사용 (텍스트 유사도를 여러 콘텐츠 변형이 공유)"""
    memo = {}

    def g(a, b):
        key = (a.size, hash(a.tobytes()), hash(b.tobytes()))
        if key not in memo:
            memo[key] = f(a, b)
        return memo[key]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test-users", type=int, default=20_000, help="Top-N 평가에 쓰는 test 사용자 수")
    ap.add_argument("--n-val-users", type=int, default=5_000, help="λ 선택에 쓰는 val 사용자 수")
    ap.add_argument("--n-neg", type=int, default=99, help="정답 1개와 함께 섞는 후보 호텔 수")
    ap.add_argument("--n-city-neg", type=int, default=49, help="같은 도시 평가에서 정답과 섞는 후보 호텔 수")
    ap.add_argument("--knn-k", type=int, default=2000, help="호텔별 이웃 수 (07_tune_itemknn.py로 선택)")
    ap.add_argument("--knn-shrink", type=float, default=100.0)
    ap.add_argument("--knn-damp", type=float, default=0.03)
    ap.add_argument("--mf-k", type=int, default=32)
    ap.add_argument("--mf-lr", type=float, default=0.01)
    ap.add_argument("--mf-reg", type=float, default=0.05)
    ap.add_argument("--mf-epochs", type=int, default=40)
    args = ap.parse_args()
    ensure_dirs()
    plt = setup_plot()

    df, n_users, n_hotels = load_split()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    u, i, r = arrays(tr)
    val, test = arrays(va), arrays(te)

    W = np.load(WORK_DIR / "user_weights.npy")
    w_g = np.load(WORK_DIR / "global_weights.npy")
    persona = pd.read_csv(WORK_DIR / "user_personas.csv").persona.to_numpy()
    A = np.load(WORK_DIR / "hotel_aspects.npy")
    T = sp.load_npz(WORK_DIR / "hotel_tfidf.npz")

    # 04·05단계 편차 기반 선호: 관대함, aspect별 까다로움(개인, train만으로 계산), 페르소나
    dev = pd.read_csv(WORK_DIR / "user_personas_dev_tuned.csv").sort_values("user_id")
    assert len(dev) == n_users and (dev.user_id.to_numpy() == np.arange(n_users)).all()
    leni = dev.leniency.to_numpy()
    strict = dev[[f"strict_{a}" for a in ASPECTS]].to_numpy()
    persona_dev = dev.persona.to_numpy()
    strict_p = pd.DataFrame(strict).groupby(persona_dev).transform("mean").to_numpy()   # 페르소나 평균 까다로움
    cold = np.bincount(u, minlength=n_users) < COLD_MAX
    strict_c = np.where(cold[:, None], strict_p, strict)
    print(f"편차 기반 선호 로드: 페르소나 {persona_dev.max() + 1}개, 이력이 적은 사용자(train {COLD_MAX}건 미만) "
          f"{cold.mean():.1%}")

    # ------------------------------------------------------------ 학습
    print("[1/4] 모델 학습")
    times = {}

    def timed(name, f):
        t0 = time.time()
        m = f()
        times[name] = round(time.time() - t0, 1)
        return m

    base = timed("Baseline", lambda: BiasBaseline().fit(u, i, r, n_users, n_hotels))
    pop = timed("Popularity", lambda: Popularity().fit(u, i, r, n_users, n_hotels))
    ppop = timed("PersonaPopularity(회귀)", lambda: PersonaPopularity(persona).fit(u, i, r, n_users, n_hotels))
    ppop_dev = timed("PersonaPopularity(편차)", lambda: PersonaPopularity(persona_dev).fit(u, i, r, n_users, n_hotels))
    knn = timed("ItemKNN", lambda: ItemKNN(k=args.knn_k, shrink=args.knn_shrink, damp=args.knn_damp)
                .fit(u, i, r, n_users, n_hotels))
    # 비교용 기존판: 사용자 평균 기준(adjusted cosine), 축소 약하게, 분모 축소 없음
    knn_old = timed("ItemKNN(기존)", lambda: ItemKNN(k=50, shrink=10, damp=0, center="user_mean")
                    .fit(u, i, r, n_users, n_hotels))
    mf = timed("MF", lambda: BiasedMF(k=args.mf_k, lr=args.mf_lr, reg=args.mf_reg, epochs=args.mf_epochs)
               .fit(u, i, r, n_users, n_hotels, val=val))
    cb = ContentBased(W, A, T)
    cb.text_sim = cache_pairs(cb.text_sim)          # 텍스트 유사도는 아래 콘텐츠 변형들이 함께 재사용
    cb = timed("Content", lambda: cb.fit(u, i, r, n_users, n_hotels).fit_gamma(*val))
    # 비교용: 개인 가중치 대신 모든 사용자에게 전역 가중치를 쓴 콘텐츠 기반 (aspect 효용만)
    cb_g = ContentBased(np.tile(w_g, (n_users, 1)), A, T)

    hist = pd.DataFrame(mf.history)
    save_table(hist, "06_mf_history")
    plt.figure(figsize=(6, 4))
    plt.plot(hist.epoch, hist.train_rmse, marker="o", label="train")
    plt.plot(hist.epoch, hist.val_rmse, marker="o", label="val")
    plt.axvline(mf.best_epoch, ls="--", color="gray")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("MF 학습 곡선 (점선: 선택된 epoch)")
    plt.legend()
    save_fig(plt, "06_mf_learning_curve")

    rating_models = {
        "GlobalMean": lambda a, b: np.full(len(a), r.mean()),
        "Baseline(μ+b_u+b_i)": base.predict,
        "ItemKNN(기존)": knn_old.predict,
        "ItemKNN": knn.predict,
        "MF": mf.predict,
        "Content(aspect, 전역 w)": lambda a, b: np.clip(cb_g.utility(a, b), 1, 5),
        "Content(aspect, 개인 w)": lambda a, b: np.clip(cb.utility(a, b), 1, 5),
        "Content(aspect+text)": cb.predict,
    }
    rank_models = {"Popularity": pop.predict, "PersonaPopularity(회귀)": ppop.predict,
                   "PersonaPopularity(편차)": ppop_dev.predict,
                   **{k: v for k, v in rating_models.items() if k != "GlobalMean"}}
    rank_models["ItemKNN"] = knn.rank_score          # 개선판은 순위용 점수(공동 방문 유사도 합)를 따로 씀

    # ------------------------------------------------------------ 후보 생성
    print("[2/4] Top-N 평가 후보 생성")
    rng = np.random.default_rng(SEED)

    def make_candidates(part, n):
        users = np.sort(rng.choice(part.user_id.unique(), size=min(n, part.user_id.nunique()), replace=False))
        pos = part.set_index("user_id").loc[users, "hotel_id"].to_numpy()
        return users, sample_candidates(df, users, pos, n_hotels, args.n_neg, rng)

    val_users, val_cand = make_candidates(va, args.n_val_users)
    test_users, test_cand = make_candidates(te, args.n_test_users)

    def scores(f, users, cand):
        return f(np.repeat(users, cand.shape[1]), cand.ravel()).reshape(cand.shape)

    # ------------------------------------------------------------ 편차 기반 선호 연결 (val로 τ, κ 선택)
    print("[3/4] 편차 기반 선호 연결(τ, κ)과 하이브리드 λ 선택 (val)")
    vc_u, vc_i = np.repeat(val_users, val_cand.shape[1]), val_cand.ravel()
    v_sim = cb.text_sim(vc_u, vc_i).reshape(val_cand.shape)
    tau_rows = []
    for tau in TAUS:
        util = ContentBased(strict_weights(w_g, strict, tau), A, T).utility(vc_u, vc_i).reshape(val_cand.shape)
        met, _ = ranking_metrics(util + cb.gamma * (v_sim - cb.s_mean))
        tau_rows.append({"tau": tau, **met})
    tau_tab = pd.DataFrame(tau_rows)
    tau = float(tau_tab.loc[tau_tab["NDCG@10"].idxmax(), "tau"])
    # κ: 관대함으로 val 평점 잔차를 설명하는 계수 (절편 없는 최소제곱 = Σ L·잔차 / Σ L²)
    util_v = (ContentBased(strict_weights(w_g, strict, tau), A, T).utility(val[0], val[1])
              + cb.gamma * (cb.text_sim(val[0], val[1]) - cb.s_mean))
    Lv = leni[val[0]]
    kappa = float((Lv * (val[2] - util_v)).sum() / (Lv * Lv).sum())
    tau_tab["selected"] = tau_tab.tau == tau
    tau_tab["kappa"] = kappa
    print(tau_tab.round(4).to_string(index=False))
    print(f"   선택 τ = {tau}, 관대함 계수 κ = {kappa:.3f}")
    save_table(tau_tab, "06_strictness_tau")

    def content_strict(S):
        """까다로움 S를 반영한 콘텐츠 기반 (텍스트 프로필·γ는 기존 콘텐츠 기반과 공유)"""
        m = ContentBased(strict_weights(w_g, S, tau, leni, kappa), A, T)
        m.R, m.gamma, m.s_mean, m.text_sim = cb.R, cb.gamma, cb.s_mean, cb.text_sim
        return m

    cbs = {"Content(까다로움, 개인)": content_strict(strict),
           "Content(까다로움, 페르소나)": content_strict(strict_p),
           "Content(까다로움, 이력 적으면 페르소나)": content_strict(strict_c)}
    for name, m in cbs.items():
        rating_models[name] = rank_models[name] = m.predict
    cb_cold = cbs["Content(까다로움, 이력 적으면 페르소나)"]

    # ------------------------------------------------------------ 하이브리드 λ 선택 (val)
    v_mf = scores(mf.predict, val_users, val_cand)
    vp_mf = mf.predict(val[0], val[1])

    def choose_lambda(content, label):
        v_c, vp_c = scores(content.predict, val_users, val_cand), content.predict(val[0], val[1])
        rows = []
        for lm in LAMBDAS:
            met, _ = ranking_metrics(lm * v_mf + (1 - lm) * v_c)
            rows.append({"lambda": lm, "val_rmse": rmse(val[2], lm * vp_mf + (1 - lm) * vp_c), **met})
        tab = pd.DataFrame(rows)
        best = float(tab.loc[tab["NDCG@10"].idxmax(), "lambda"])
        print(f"   {label}: 선택 λ = {best}  (λ=1: MF만, λ=0: 콘텐츠 기반만)")
        return tab, best

    lam_tab, lam = choose_lambda(cb, "Hybrid (MF + Content)")
    print(lam_tab.round(4).to_string(index=False))
    save_table(lam_tab, "06_hybrid_lambda")
    lam_tab_s, lam_s = choose_lambda(cb_cold, "Hybrid (MF + Content 까다로움)")
    save_table(lam_tab_s, "06_hybrid_lambda_strict")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(lam_tab["lambda"], lam_tab.val_rmse, marker="o")
    ax[0].set_title("val RMSE")
    ax[1].plot(lam_tab["lambda"], lam_tab["NDCG@10"], marker="o", color="C1")
    ax[1].set_title("val NDCG@10")
    for a in ax:
        a.axvline(lam, ls="--", color="gray")
        a.set_xlabel("λ (MF 비중)")
    save_fig(plt, "06_hybrid_lambda")

    hybrid = lambda a, b: lam * mf.predict(a, b) + (1 - lam) * cb.predict(a, b)
    rating_models[f"Hybrid(λ={lam})"] = hybrid
    rank_models[f"Hybrid(λ={lam})"] = hybrid
    hybrid_s = lambda a, b: lam_s * mf.predict(a, b) + (1 - lam_s) * cb_cold.predict(a, b)
    rating_models[f"Hybrid(까다로움, λ={lam_s})"] = hybrid_s
    rank_models[f"Hybrid(까다로움, λ={lam_s})"] = hybrid_s

    # 11단계(Kaggle 호텔 매칭)가 쓰는 호텔 단위 점수: 신규 사용자 예측은 사용자 항이 0
    np.savez(WORK_DIR / "hotel_scores.npz", base_mu=base.mu, base_bi=base.bi, mf_mu=mf.mu, mf_bi=mf.bi,
             hybrid_lambda=lam)

    # ------------------------------------------------------------ test 평가
    print("[4/4] test 평가")
    rows = []
    for name, f in rating_models.items():
        pv, pt = f(val[0], val[1]), f(test[0], test[1])
        rows.append({"model": name, "val_RMSE": rmse(val[2], pv), "val_MAE": mae(val[2], pv),
                     "test_RMSE": rmse(test[2], pt), "test_MAE": mae(test[2], pt),
                     "train_time_s": times.get(name, times.get(name.split("(")[0], ""))})
    rating_tab = pd.DataFrame(rows)
    print(rating_tab.round(4).to_string(index=False))
    save_table(rating_tab, "06_rating_prediction")

    n_train = np.bincount(u, minlength=n_users)[test_users]
    bucket = pd.cut(n_train, ACTIVITY_BINS, right=False).astype(str)
    rows, act_rows = [], []
    for name, f in rank_models.items():
        met, rank = ranking_metrics(scores(f, test_users, test_cand))
        rows.append({"model": name, **met})
        for b in pd.unique(bucket):
            m = bucket == b
            act_rows.append({"model": name, "train_reviews": b, "n_users": int(m.sum()),
                             "HR@10": float((rank[m] < 10).mean())})
    rank_tab = pd.DataFrame(rows)
    print(f"Top-N (test 사용자 {len(test_users):,}명, 후보 1+{args.n_neg})")
    print(rank_tab.round(4).to_string(index=False))
    save_table(rank_tab, "06_topn_ranking")
    act = pd.DataFrame(act_rows)
    save_table(act, "06_topn_by_activity")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].barh(rank_tab.model, rank_tab["NDCG@10"], color="#4c72b0")
    ax[0].invert_yaxis()
    ax[0].set_title("test NDCG@10")
    piv = act.pivot(index="train_reviews", columns="model", values="HR@10")[ACTIVITY_PLOT]   # 주요 모델만 그림
    piv = piv.loc[sorted(piv.index, key=lambda s: float(s[1:].split(",")[0]))]
    piv.plot(ax=ax[1], marker="o")
    ax[1].set_title("사용자 train 리뷰 수별 test HR@10")
    ax[1].set_xlabel("train 리뷰 수 구간")
    ax[1].legend(fontsize=7, ncol=2)
    save_fig(plt, "06_topn_results")

    # ------------------------------------------------------------ 같은 도시 후보 Top-N
    hm = pd.read_csv(HOTELREC_DIR / "hotel_map.csv", encoding="utf-8-sig").sort_values("hotel_id")
    if len(hm) != n_hotels or not (hm.hotel_id.to_numpy() == np.arange(n_hotels)).all():
        print(f"[건너뜀] hotel_map.csv({len(hm):,}행)가 리뷰 데이터의 호텔 수({n_hotels:,}개)와 맞지 않아 "
              "같은 도시 평가를 하지 않습니다")
        return
    city = hm.location.fillna("(unknown)").to_numpy()
    pos = te.set_index("user_id").loc[test_users, "hotel_id"].to_numpy()
    c_users, c_cand = sample_city_candidates(df, test_users, pos, city, args.n_city_neg, rng)
    print(f"같은 도시 Top-N: test 사용자 {len(test_users):,}명 중 같은 도시 후보가 {args.n_city_neg}개 이상인 "
          f"{len(c_users):,}명 (후보 1+{args.n_city_neg})")
    rows = []
    for name, f in rank_models.items():
        met, _ = ranking_metrics(scores(f, c_users, c_cand))
        rows.append({"model": name, **met})
    city_tab = pd.DataFrame(rows)
    print(city_tab.round(4).to_string(index=False))
    save_table(city_tab, "06_topn_same_city")

    both = rank_tab[["model", "NDCG@10"]].merge(city_tab[["model", "NDCG@10"]], on="model",
                                                 suffixes=(" (무작위 후보)", " (같은 도시 후보)"))
    ax = both.set_index("model").plot.barh(figsize=(8, 5))
    ax.invert_yaxis()
    ax.set_title("test NDCG@10: 후보 구성에 따른 비교")
    save_fig(plt, "06_topn_same_city")


if __name__ == "__main__":
    main()
