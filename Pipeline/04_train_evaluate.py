"""
4단계: 모델 학습과 평가 (HotelRec)

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

하이브리드 (제안서 ④):  score = λ · MF + (1 - λ) · ContentBased,  λ는 val NDCG@10이 가장 높은 값

출력: Pipeline/outputs/04_*.csv, figures/04_*.png
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import (SEED, WORK_DIR, ensure_dirs, load_split, mae, ranking_metrics, rmse,
                    sample_candidates, save_fig, save_table, setup_plot)
from models import BiasBaseline, BiasedMF, ContentBased, ItemKNN, PersonaPopularity, Popularity

LAMBDAS = np.round(np.arange(0, 1.01, 0.1), 1)
ACTIVITY_BINS = [3, 5, 10, 20, np.inf]


def arrays(df):
    return df.user_id.to_numpy(), df.hotel_id.to_numpy(), df.rating.to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test-users", type=int, default=20_000, help="Top-N 평가에 쓰는 test 사용자 수")
    ap.add_argument("--n-val-users", type=int, default=5_000, help="λ 선택에 쓰는 val 사용자 수")
    ap.add_argument("--n-neg", type=int, default=99, help="정답 1개와 함께 섞는 후보 호텔 수")
    ap.add_argument("--knn-k", type=int, default=50)
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
    ppop = timed("PersonaPopularity", lambda: PersonaPopularity(persona).fit(u, i, r, n_users, n_hotels))
    knn = timed("ItemKNN", lambda: ItemKNN(k=args.knn_k).fit(u, i, r, n_users, n_hotels))
    mf = timed("MF", lambda: BiasedMF(k=args.mf_k, lr=args.mf_lr, reg=args.mf_reg, epochs=args.mf_epochs)
               .fit(u, i, r, n_users, n_hotels, val=val))
    cb = timed("Content", lambda: ContentBased(W, A, T).fit(u, i, r, n_users, n_hotels).fit_gamma(*val))
    # 비교용: 개인 가중치 대신 모든 사용자에게 전역 가중치를 쓴 콘텐츠 기반 (aspect 효용만)
    cb_g = ContentBased(np.tile(w_g, (n_users, 1)), A, T)

    hist = pd.DataFrame(mf.history)
    save_table(hist, "04_mf_history")
    plt.figure(figsize=(6, 4))
    plt.plot(hist.epoch, hist.train_rmse, marker="o", label="train")
    plt.plot(hist.epoch, hist.val_rmse, marker="o", label="val")
    plt.axvline(mf.best_epoch, ls="--", color="gray")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("MF 학습 곡선 (점선: 선택된 epoch)")
    plt.legend()
    save_fig(plt, "04_mf_learning_curve")

    rating_models = {
        "GlobalMean": lambda a, b: np.full(len(a), r.mean()),
        "Baseline(μ+b_u+b_i)": base.predict,
        "ItemKNN": knn.predict,
        "MF": mf.predict,
        "Content(aspect, 전역 w)": lambda a, b: np.clip(cb_g.utility(a, b), 1, 5),
        "Content(aspect, 개인 w)": lambda a, b: np.clip(cb.utility(a, b), 1, 5),
        "Content(aspect+text)": cb.predict,
    }
    rank_models = {"Popularity": pop.predict, "PersonaPopularity": ppop.predict, **{
        k: v for k, v in rating_models.items() if k != "GlobalMean"}}

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

    # ------------------------------------------------------------ 하이브리드 λ 선택 (val)
    print("[3/4] 하이브리드 λ 선택 (val)")
    v_mf, v_cb = scores(mf.predict, val_users, val_cand), scores(cb.predict, val_users, val_cand)
    vp_mf, vp_cb = mf.predict(val[0], val[1]), cb.predict(val[0], val[1])
    lam_rows = []
    for lam in LAMBDAS:
        met, _ = ranking_metrics(lam * v_mf + (1 - lam) * v_cb)
        lam_rows.append({"lambda": lam, "val_rmse": rmse(val[2], lam * vp_mf + (1 - lam) * vp_cb), **met})
    lam_tab = pd.DataFrame(lam_rows)
    print(lam_tab.round(4).to_string(index=False))
    lam = float(lam_tab.loc[lam_tab["NDCG@10"].idxmax(), "lambda"])
    print(f"   선택 λ = {lam}  (λ=1: MF만, λ=0: 콘텐츠 기반만)")
    save_table(lam_tab, "04_hybrid_lambda")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(lam_tab["lambda"], lam_tab.val_rmse, marker="o")
    ax[0].set_title("val RMSE")
    ax[1].plot(lam_tab["lambda"], lam_tab["NDCG@10"], marker="o", color="C1")
    ax[1].set_title("val NDCG@10")
    for a in ax:
        a.axvline(lam, ls="--", color="gray")
        a.set_xlabel("λ (MF 비중)")
    save_fig(plt, "04_hybrid_lambda")

    hybrid = lambda a, b: lam * mf.predict(a, b) + (1 - lam) * cb.predict(a, b)
    rating_models[f"Hybrid(λ={lam})"] = hybrid
    rank_models[f"Hybrid(λ={lam})"] = hybrid

    # ------------------------------------------------------------ test 평가
    print("[4/4] test 평가")
    rows = []
    for name, f in rating_models.items():
        pv, pt = f(val[0], val[1]), f(test[0], test[1])
        rows.append({"model": name, "val_RMSE": rmse(val[2], pv), "val_MAE": mae(val[2], pv),
                     "test_RMSE": rmse(test[2], pt), "test_MAE": mae(test[2], pt),
                     "train_time_s": times.get(name.split("(")[0], "")})
    rating_tab = pd.DataFrame(rows)
    print(rating_tab.round(4).to_string(index=False))
    save_table(rating_tab, "04_rating_prediction")

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
    save_table(rank_tab, "04_topn_ranking")
    act = pd.DataFrame(act_rows)
    save_table(act, "04_topn_by_activity")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].barh(rank_tab.model, rank_tab["NDCG@10"], color="#4c72b0")
    ax[0].invert_yaxis()
    ax[0].set_title("test NDCG@10")
    piv = act.pivot(index="train_reviews", columns="model", values="HR@10")
    piv = piv.loc[sorted(piv.index, key=lambda s: float(s[1:].split(",")[0]))]
    piv.plot(ax=ax[1], marker="o")
    ax[1].set_title("사용자 train 리뷰 수별 test HR@10")
    ax[1].set_xlabel("train 리뷰 수 구간")
    ax[1].legend(fontsize=7, ncol=2)
    save_fig(plt, "04_topn_results")


if __name__ == "__main__":
    main()
