"""
Step 6: model training and evaluation (HotelRec)

Training:  all models are trained on the train split.
Selection: the val split is used for MF early stopping, the content model's γ and the hybrid weights λ.
Evaluation: the test split (each user's most recent review)

(1) Rating prediction: RMSE and MAE over every user's val/test review
(2) Top-N recommendation: for randomly drawn users, rank 1 ground-truth hotel (the hotel of the test review)
    + N_NEG hotels the user never rated by score, then compute HR@k and NDCG@k
    (a sampled-candidate protocol instead of ranking all 47k hotels; since candidates are random, the
     comparison between models matters more than the absolute values.)
(3) HR@10 by user activity (number of train reviews): does content-based / hybrid help users with few reviews?
    (the motivation in section 1 of the proposal)
(4) Same-city Top-N: candidates are drawn only from the ground-truth hotel's city to evaluate
    "choosing a hotel within a destination"
    (with random candidates a popular hotel only has to be told apart from unpopular hotels elsewhere,
     which strongly favors the popularity baseline)
    Runs only when hotel_map.csv uses the same IDs as the review data.

Item-kNN is reported in both the improved version (KNNBaseline + damped denominator + co-visit ranking score)
and the original version (adjusted cosine).

Hybrid (proposal ④: weighted sum of a CF score and a content score), split into two with different goals.
    Hybrid-rating (MF+Content):      λ · MF + (1 - λ) · Content(strictness, persona if few reviews)
                                     both on the rating scale, summed directly; λ minimizes val RMSE
    Hybrid-ranking (ItemKNN+Content): λ · z(ItemKNN co-visit score) + (1 - λ) · z(Content(strictness, persona if few reviews))
                                     z: standardized among the same user's candidates (the two scores use different scales)
                                     λ maximizes val NDCG@10. Uses ItemKNN, the strongest CF for ranking.
    (Choosing λ for MF + Content by val NDCG@10 gave λ=0, i.e. the same as content-based, hence the split.)

Connecting deviation-based preferences (user_personas_dev_tuned.csv from steps 04-05)
    PersonaPopularity(deviation): recommends hotels rated 4+ by users with the same persona (strictness type) from 05
    Content(strictness): tilts the content model's aspect weights toward the aspects the user is strict about.
        w_uk = w_g,k · exp(τ · strict_uk), normalized so that Σ_k w_uk = Σ_k w_g,k (keeps the rating scale)
        intercept = w_g,0 + κ · leniency_u   (leniency is constant per user, so it changes ratings, not rankings)
        τ is chosen by val NDCG@10, κ by least squares on the val rating residuals.
        Strictness is compared three ways: personal / persona mean / persona only for users with few reviews
        (fewer than COLD_MAX train reviews). The last is the role table's "fall back to persona weights for
        cold start" and is used as the content side of both hybrids.

Output: Pipeline/outputs/06_*.csv, figures/06_*.png
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
# Candidate strengths for strictness. Strictness is mostly within ±0.1 rating points, so τ must be in the
# tens to hundreds to change the weights.
# (median weight share of the strictest aspect: τ=80 → 0.66, 160 → 0.89, 320 → 0.99)
TAUS = [0, 5, 10, 20, 40, 80, 160, 320]
COLD_MAX = 5                        # users with fewer train reviews than this use persona strictness instead of personal
HYB_RATING, HYB_RANK = "Hybrid-rating (MF+Content)", "Hybrid-ranking (ItemKNN+Content)"
ACTIVITY_PLOT = ["Popularity", "ItemKNN", "MF", "Content(strictness, persona if few reviews)", HYB_RANK]


def arrays(df):
    return df.user_id.to_numpy(), df.hotel_id.to_numpy(), df.rating.to_numpy(float)


def strict_weights(w_g, S, tau, leniency=None, kappa=0.0):
    """
    Per-user [intercept, 6 aspect weights] reflecting strictness (same format as ContentBased's W)
        w_uk ∝ w_g,k · exp(τ · S_uk), rescaled to the same total as the global weights / intercept = w_g,0 + κ · leniency
    """
    w = w_g[1:] * np.exp(tau * S)
    w *= w_g[1:].sum() / w.sum(1, keepdims=True)
    w0 = w_g[0] + (kappa * leniency if leniency is not None else 0.0)
    return np.column_stack([np.broadcast_to(w0, len(S)), w])


def zscore_by_user(u, s):
    """Standardize scores among the same user's (candidates) to mean 0, std 1. Users whose scores are all equal get 0."""
    n = np.maximum(np.bincount(u), 1)
    mean = np.bincount(u, s) / n
    sd = np.sqrt(np.maximum(np.bincount(u, s * s) / n - mean ** 2, 0))
    return np.divide(s - mean[u], sd[u], out=np.zeros_like(s, dtype=float), where=sd[u] > 1e-12)


def cache_pairs(f):
    """Store and reuse results for the same (user, hotel) arrays (the content variants share the text similarity)"""
    memo = {}

    def g(a, b):
        key = (a.size, hash(a.tobytes()), hash(b.tobytes()))
        if key not in memo:
            memo[key] = f(a, b)
        return memo[key]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test-users", type=int, default=20_000, help="number of test users for Top-N evaluation")
    ap.add_argument("--n-val-users", type=int, default=5_000, help="number of val users for choosing λ")
    ap.add_argument("--n-neg", type=int, default=99, help="number of candidate hotels mixed with the ground truth")
    ap.add_argument("--n-city-neg", type=int, default=49, help="number of candidates mixed with the ground truth in the same-city evaluation")
    ap.add_argument("--knn-k", type=int, default=2000, help="number of neighbors per hotel (chosen with 07_tune_itemknn.py)")
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

    # deviation-based preferences from steps 04-05: leniency, per-aspect strictness (personal, train only), persona
    dev = pd.read_csv(WORK_DIR / "user_personas_dev_tuned.csv").sort_values("user_id")
    assert len(dev) == n_users and (dev.user_id.to_numpy() == np.arange(n_users)).all()
    leni = dev.leniency.to_numpy()
    strict = dev[[f"strict_{a}" for a in ASPECTS]].to_numpy()
    persona_dev = dev.persona.to_numpy()
    strict_p = pd.DataFrame(strict).groupby(persona_dev).transform("mean").to_numpy()   # persona-mean strictness
    cold = np.bincount(u, minlength=n_users) < COLD_MAX
    strict_c = np.where(cold[:, None], strict_p, strict)
    print(f"loaded deviation-based preferences: {persona_dev.max() + 1} personas, users with few reviews "
          f"(fewer than {COLD_MAX} train reviews) {cold.mean():.1%}")

    # ------------------------------------------------------------ training
    print("[1/4] training models")
    times = {}

    def timed(name, f):
        t0 = time.time()
        m = f()
        times[name] = round(time.time() - t0, 1)
        return m

    base = timed("Baseline", lambda: BiasBaseline().fit(u, i, r, n_users, n_hotels))
    pop = timed("Popularity", lambda: Popularity().fit(u, i, r, n_users, n_hotels))
    ppop = timed("PersonaPopularity(regression)", lambda: PersonaPopularity(persona).fit(u, i, r, n_users, n_hotels))
    ppop_dev = timed("PersonaPopularity(deviation)", lambda: PersonaPopularity(persona_dev).fit(u, i, r, n_users, n_hotels))
    knn = timed("ItemKNN", lambda: ItemKNN(k=args.knn_k, shrink=args.knn_shrink, damp=args.knn_damp)
                .fit(u, i, r, n_users, n_hotels))
    # original version for comparison: centered on the user mean (adjusted cosine), weak shrinkage, no damping
    knn_old = timed("ItemKNN(original)", lambda: ItemKNN(k=50, shrink=10, damp=0, center="user_mean")
                    .fit(u, i, r, n_users, n_hotels))
    mf = timed("MF", lambda: BiasedMF(k=args.mf_k, lr=args.mf_lr, reg=args.mf_reg, epochs=args.mf_epochs)
               .fit(u, i, r, n_users, n_hotels, val=val))
    cb = ContentBased(W, A, T)
    cb.text_sim = cache_pairs(cb.text_sim)          # the text similarity is reused by the content variants below
    cb = timed("Content", lambda: cb.fit(u, i, r, n_users, n_hotels).fit_gamma(*val))
    # for comparison: content-based with the global weights for everyone instead of personal weights (aspect utility only)
    cb_g = ContentBased(np.tile(w_g, (n_users, 1)), A, T)

    hist = pd.DataFrame(mf.history)
    save_table(hist, "06_mf_history")
    plt.figure(figsize=(6, 4))
    plt.plot(hist.epoch, hist.train_rmse, marker="o", label="train")
    plt.plot(hist.epoch, hist.val_rmse, marker="o", label="val")
    plt.axvline(mf.best_epoch, ls="--", color="gray")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("MF learning curve (dashed: selected epoch)")
    plt.legend()
    save_fig(plt, "06_mf_learning_curve")

    rating_models = {
        "GlobalMean": lambda a, b: np.full(len(a), r.mean()),
        "Baseline(μ+b_u+b_i)": base.predict,
        "ItemKNN(original)": knn_old.predict,
        "ItemKNN": knn.predict,
        "MF": mf.predict,
        "Content(aspect, global w)": lambda a, b: np.clip(cb_g.utility(a, b), 1, 5),
        "Content(aspect, personal w)": lambda a, b: np.clip(cb.utility(a, b), 1, 5),
        "Content(aspect+text)": cb.predict,
    }
    rank_models = {"Popularity": pop.predict, "PersonaPopularity(regression)": ppop.predict,
                   "PersonaPopularity(deviation)": ppop_dev.predict,
                   **{k: v for k, v in rating_models.items() if k != "GlobalMean"}}
    rank_models["ItemKNN"] = knn.rank_score          # the improved version uses a separate ranking score (sum of co-visit similarities)

    # ------------------------------------------------------------ candidates
    print("[2/4] building Top-N evaluation candidates")
    rng = np.random.default_rng(SEED)

    def make_candidates(part, n):
        users = np.sort(rng.choice(part.user_id.unique(), size=min(n, part.user_id.nunique()), replace=False))
        pos = part.set_index("user_id").loc[users, "hotel_id"].to_numpy()
        return users, sample_candidates(df, users, pos, n_hotels, args.n_neg, rng)

    val_users, val_cand = make_candidates(va, args.n_val_users)
    test_users, test_cand = make_candidates(te, args.n_test_users)

    def scores(f, users, cand):
        return f(np.repeat(users, cand.shape[1]), cand.ravel()).reshape(cand.shape)

    # ------------------------------------------------------------ connect deviation-based preferences (τ, κ on val)
    print("[3/4] connecting deviation-based preferences (τ, κ) and choosing hybrid λ (val)")
    vc_u, vc_i = np.repeat(val_users, val_cand.shape[1]), val_cand.ravel()
    v_sim = cb.text_sim(vc_u, vc_i).reshape(val_cand.shape)
    tau_rows = []
    for tau in TAUS:
        util = ContentBased(strict_weights(w_g, strict, tau), A, T).utility(vc_u, vc_i).reshape(val_cand.shape)
        met, _ = ranking_metrics(util + cb.gamma * (v_sim - cb.s_mean))
        tau_rows.append({"tau": tau, **met})
    tau_tab = pd.DataFrame(tau_rows)
    tau = float(tau_tab.loc[tau_tab["NDCG@10"].idxmax(), "tau"])
    # κ: coefficient explaining the val rating residuals with leniency (least squares without intercept = Σ L·resid / Σ L²)
    util_v = (ContentBased(strict_weights(w_g, strict, tau), A, T).utility(val[0], val[1])
              + cb.gamma * (cb.text_sim(val[0], val[1]) - cb.s_mean))
    Lv = leni[val[0]]
    kappa = float((Lv * (val[2] - util_v)).sum() / (Lv * Lv).sum())
    tau_tab["selected"] = tau_tab.tau == tau
    tau_tab["kappa"] = kappa
    print(tau_tab.round(4).to_string(index=False))
    print(f"   selected τ = {tau}, leniency coefficient κ = {kappa:.3f}")
    save_table(tau_tab, "06_strictness_tau")

    def content_strict(S):
        """Content-based model reflecting strictness S (text profile and γ shared with the base content model)"""
        m = ContentBased(strict_weights(w_g, S, tau, leni, kappa), A, T)
        m.R, m.gamma, m.s_mean, m.text_sim = cb.R, cb.gamma, cb.s_mean, cb.text_sim
        return m

    cbs = {"Content(strictness, personal)": content_strict(strict),
           "Content(strictness, persona)": content_strict(strict_p),
           "Content(strictness, persona if few reviews)": content_strict(strict_c)}
    for name, m in cbs.items():
        rating_models[name] = rank_models[name] = m.predict
    cb_cold = cbs["Content(strictness, persona if few reviews)"]

    # ------------------------------------------------------------ choose hybrid λ (val)
    # both hybrids use Content(strictness, persona if few reviews) as the content side.
    vp_mf, vp_c = mf.predict(val[0], val[1]), cb_cold.predict(val[0], val[1])
    v_mf, v_c = scores(mf.predict, val_users, val_cand), scores(cb_cold.predict, val_users, val_cand)
    v_knn = scores(knn.rank_score, val_users, val_cand)
    zc = lambda m: zscore_by_user(np.repeat(val_users, val_cand.shape[1]), m.ravel()).reshape(m.shape)
    z_knn, z_c = zc(v_knn), zc(v_c)

    rows = []
    for lm in LAMBDAS:
        # the rating hybrid is chosen by RMSE, but its ranking metrics are recorded too
        met_r, _ = ranking_metrics(lm * v_mf + (1 - lm) * v_c)
        met_k, _ = ranking_metrics(lm * z_knn + (1 - lm) * z_c)
        rows.append({"lambda": lm, "rating_val_RMSE": rmse(val[2], lm * vp_mf + (1 - lm) * vp_c),
                     "rating_val_NDCG@10": met_r["NDCG@10"],
                     "rank_val_HR@10": met_k["HR@10"], "rank_val_NDCG@10": met_k["NDCG@10"]})
    lam_tab = pd.DataFrame(rows)
    lam_r = float(lam_tab.loc[lam_tab.rating_val_RMSE.idxmin(), "lambda"])
    lam_k = float(lam_tab.loc[lam_tab["rank_val_NDCG@10"].idxmax(), "lambda"])
    lam_tab["selected_rating"] = lam_tab["lambda"] == lam_r
    lam_tab["selected_rank"] = lam_tab["lambda"] == lam_k
    print(lam_tab.round(4).to_string(index=False))
    print(f"   {HYB_RATING}: λ = {lam_r} (MF share, lowest val RMSE)")
    print(f"   {HYB_RANK}: λ = {lam_k} (ItemKNN share, highest val NDCG@10)")
    save_table(lam_tab, "06_hybrid_lambda")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(lam_tab["lambda"], lam_tab.rating_val_RMSE, marker="o")
    ax[0].axvline(lam_r, ls="--", color="gray")
    ax[0].set_title(f"{HYB_RATING}: val RMSE")
    ax[0].set_xlabel("λ (MF share)")
    ax[1].plot(lam_tab["lambda"], lam_tab["rank_val_NDCG@10"], marker="o", color="C1")
    ax[1].axvline(lam_k, ls="--", color="gray")
    ax[1].set_title(f"{HYB_RANK}: val NDCG@10")
    ax[1].set_xlabel("λ (ItemKNN share)")
    save_fig(plt, "06_hybrid_lambda")

    hyb_rating = lambda a, b: lam_r * mf.predict(a, b) + (1 - lam_r) * cb_cold.predict(a, b)
    hyb_rank = lambda a, b: (lam_k * zscore_by_user(a, knn.rank_score(a, b))
                             + (1 - lam_k) * zscore_by_user(a, cb_cold.predict(a, b)))
    rating_models[HYB_RATING] = rank_models[HYB_RATING] = hyb_rating
    rank_models[HYB_RANK] = hyb_rank                # its scores are standardized, so it is used for ranking only

    # hotel-level scores used by step 11 (Kaggle hotel matching): for a new user the user terms are 0 (uses the rating hybrid's λ)
    np.savez(WORK_DIR / "hotel_scores.npz", base_mu=base.mu, base_bi=base.bi, mf_mu=mf.mu, mf_bi=mf.bi,
             hybrid_lambda=lam_r)

    # ------------------------------------------------------------ test evaluation
    print("[4/4] test evaluation")
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
    print(f"Top-N ({len(test_users):,} test users, candidates 1+{args.n_neg})")
    print(rank_tab.round(4).to_string(index=False))
    save_table(rank_tab, "06_topn_ranking")
    act = pd.DataFrame(act_rows)
    save_table(act, "06_topn_by_activity")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].barh(rank_tab.model, rank_tab["NDCG@10"], color="#4c72b0")
    ax[0].invert_yaxis()
    ax[0].set_title("test NDCG@10")
    piv = act.pivot(index="train_reviews", columns="model", values="HR@10")[ACTIVITY_PLOT]   # plot only the main models
    piv = piv.loc[sorted(piv.index, key=lambda s: float(s[1:].split(",")[0]))]
    piv.plot(ax=ax[1], marker="o")
    ax[1].set_title("test HR@10 by the user's number of train reviews")
    ax[1].set_xlabel("number of train reviews")
    ax[1].legend(fontsize=7, ncol=2)
    save_fig(plt, "06_topn_results")

    # ------------------------------------------------------------ same-city Top-N
    hm = pd.read_csv(HOTELREC_DIR / "hotel_map.csv", encoding="utf-8-sig").sort_values("hotel_id")
    if len(hm) != n_hotels or not (hm.hotel_id.to_numpy() == np.arange(n_hotels)).all():
        print(f"[skipped] hotel_map.csv ({len(hm):,} rows) does not match the number of hotels in the review data "
              f"({n_hotels:,}), so the same-city evaluation is not run")
        return
    city = hm.location.fillna("(unknown)").to_numpy()
    pos = te.set_index("user_id").loc[test_users, "hotel_id"].to_numpy()
    c_users, c_cand = sample_city_candidates(df, test_users, pos, city, args.n_city_neg, rng)
    print(f"same-city Top-N: {len(c_users):,} of {len(test_users):,} test users have {args.n_city_neg}+ same-city "
          f"candidates (candidates 1+{args.n_city_neg})")
    rows = []
    for name, f in rank_models.items():
        met, _ = ranking_metrics(scores(f, c_users, c_cand))
        rows.append({"model": name, **met})
    city_tab = pd.DataFrame(rows)
    print(city_tab.round(4).to_string(index=False))
    save_table(city_tab, "06_topn_same_city")

    both = rank_tab[["model", "NDCG@10"]].merge(city_tab[["model", "NDCG@10"]], on="model",
                                                 suffixes=(" (random candidates)", " (same-city candidates)"))
    ax = both.set_index("model").plot.barh(figsize=(8, 5))
    ax.invert_yaxis()
    ax.set_title("test NDCG@10 by candidate set")
    save_fig(plt, "06_topn_same_city")


if __name__ == "__main__":
    main()
