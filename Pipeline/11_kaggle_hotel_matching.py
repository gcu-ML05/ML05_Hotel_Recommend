"""
Step 11: HotelRec <-> Kaggle hotel matching evaluation (generalization to new users)

Kaggle (Booking.com) reviewers are new users with no history in HotelRec. For hotels present in both datasets,
check how well the scores that the HotelRec-trained models predict for a new user match the actual Booking.com scores.

(1) Hotel matching
    Within the same city, normalize hotel names (lower case, symbols removed, common words such as 'hotel' dropped)
    and match on equality. Otherwise match the most similar same-city candidate whose string similarity (difflib)
    is at least MIN_RATIO.
    The city of a HotelRec hotel is the single location in hotel_map.csv (e.g. "London England") that starts with
    the Kaggle city name and has the most hotels.

(2) New-user prediction (no user history, so the user terms are 0)
    GlobalMean          μ
    Baseline            μ + b_i
    MF                  μ + b_i                 (no latent factor p_u, so only up to the hotel bias)
    Content(aspect)     w0 + Σ_k w_k · a_hk      (global aspect weights x hotel aspect means)
    Hybrid-rating       λ · MF + (1-λ) · Content (λ is the rating hybrid's value chosen by val RMSE in step 06)

(3) Evaluation
    Review level: RMSE, MAE and Spearman rank correlation against the Kaggle review score (rating = 10-point score / 2)
    Hotel level: Pearson/Spearman correlation with each hotel's mean Kaggle score (averaging out reviewer differences)
    Review-level Spearman correlation per traveler type

Input: data/pipeline/hotel_scores.npz (step 06 output), hotel_aspects.npy, global_weights.npy
Output: Pipeline/outputs/11_*.csv, figures/11_*.png
"""
import difflib
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import HOTELREC_DIR, KAGGLE_DIR, WORK_DIR, ensure_dirs, mae, rmse, save_fig, save_table, setup_plot

MIN_RATIO = 0.9
STOP = {"hotel", "hotels", "the", "and", "by", "a", "de", "the", "&", "spa", "resort", "suites", "apartments"}


def norm_name(s):
    words = re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split()
    return " ".join(w for w in words if w not in STOP)


def match_hotels(hm, km):
    """Returns: kaggle hotel_id, hotelrec hotel_id, match type, similarity"""
    rows = []
    for city, kc in km.groupby("city"):
        hc = hm[hm.location.str.match(rf"^{city}\b", case=False, na=False)]
        if hc.empty:
            continue
        # use only the location with the most hotels, to drop different cities that share the name, like "London Ontario" or "Paris Tennessee"
        hc = hc[hc.location == hc.location.value_counts().index[0]]
        by_name = dict(zip(hc.key, hc.hotel_id))
        keys = list(by_name)
        for _, k in kc.iterrows():
            if k.key in by_name:
                rows.append((k.hotel_id, by_name[k.key], "exact", 1.0))
                continue
            # difflib.get_close_matches(string, candidates, n=max count, cutoff=min similarity): find similar strings
            best = difflib.get_close_matches(k.key, keys, n=1, cutoff=MIN_RATIO)
            if best:
                ratio = difflib.SequenceMatcher(None, k.key, best[0]).ratio()
                rows.append((k.hotel_id, by_name[best[0]], "fuzzy", ratio))
    return pd.DataFrame(rows, columns=["kaggle_hotel_id", "hotelrec_hotel_id", "match", "ratio"])


def main():
    ensure_dirs()
    plt = setup_plot()
    hm = pd.read_csv(HOTELREC_DIR / "hotel_map.csv", encoding="utf-8-sig")
    km = pd.read_csv(KAGGLE_DIR / "hotel_meta_kaggle.csv", encoding="utf-8-sig")
    k = pd.read_csv(KAGGLE_DIR / "kaggle_test.csv", encoding="utf-8-sig",
                    usecols=["hotel_id", "rating", "traveler_type"])
    hm["key"], km["key"] = hm.hotel_name.map(norm_name), km.hotel_name.map(norm_name)

    # (1) matching
    m = match_hotels(hm, km)
    # if several Kaggle hotels map to one HotelRec hotel, keep only the most similar one;
    # if the best similarity is tied (e.g. "Hotel de Nell" and "Nell Hotel Suites" both become "nell"), which one is right is unknown, so drop them all
    top = m.groupby("hotelrec_hotel_id").ratio.transform("max")
    m = m[m.ratio == top]
    n_amb = int(m.hotelrec_hotel_id.duplicated(keep=False).sum())
    m = m[~m.hotelrec_hotel_id.duplicated(keep=False)]
    print(f"Ambiguous matches dropped: {n_amb} Kaggle hotels")
    print(f"Matched: {len(m):,} of {len(km):,} Kaggle hotels (exact {(m.match == 'exact').sum()}, "
          f"fuzzy {(m.match == 'fuzzy').sum()})")
    ex = m.merge(km[["hotel_id", "hotel_name", "city"]], left_on="kaggle_hotel_id", right_on="hotel_id") \
          .merge(hm[["hotel_id", "hotel_name", "location"]], left_on="hotelrec_hotel_id", right_on="hotel_id",
                 suffixes=("_kaggle", "_hotelrec"))
    ex = ex[["kaggle_hotel_id", "hotel_name_kaggle", "city", "hotelrec_hotel_id", "hotel_name_hotelrec",
             "location", "match", "ratio"]]
    save_table(ex.sort_values(["city", "hotel_name_kaggle"]), "11_matched_hotels")
    print("Fuzzy match examples (for review):")
    print(ex[ex.match == "fuzzy"].head(10)[["hotel_name_kaggle", "hotel_name_hotelrec", "ratio"]]
          .round(3).to_string(index=False))

    # (2) new-user prediction
    s = np.load(WORK_DIR / "hotel_scores.npz")
    A = np.load(WORK_DIR / "hotel_aspects.npy")
    w_g = np.load(WORK_DIR / "global_weights.npy")
    lam = float(s["hybrid_lambda"])
    cb = w_g[0] + A @ w_g[1:]
    preds = {
        "GlobalMean": np.full(len(A), float(s["base_mu"])),
        "Baseline(μ+b_i)": s["base_mu"] + s["base_bi"],
        "MF(μ+b_i)": s["mf_mu"] + s["mf_bi"],
        "Content(aspect, global w)": cb,
        f"Hybrid-rating (MF+Content, λ={lam})": lam * (s["mf_mu"] + s["mf_bi"]) + (1 - lam) * cb,
    }
    preds = {name: np.clip(p, 1, 5) for name, p in preds.items()}

    # (3) evaluation
    to_hr = dict(zip(m.kaggle_hotel_id, m.hotelrec_hotel_id))
    kr = k[k.hotel_id.isin(to_hr)].copy()
    kr["hr_id"] = kr.hotel_id.map(to_hr)
    print(f"Kaggle reviews of matched hotels: {len(kr):,} / total {len(k):,}")
    hotel_mean = kr.groupby("hr_id").rating.mean()
    rows, type_rows = [], []
    for name, p in preds.items():
        pr = p[kr.hr_id]
        ph = p[hotel_mean.index]
        rows.append({"model": name, "review_RMSE": rmse(kr.rating, pr), "review_MAE": mae(kr.rating, pr),
                     # a constant prediction (GlobalMean) has no ranking, so the correlation is undefined (NaN)
                     "review_spearman": spearmanr(pr, kr.rating).statistic if pr.std() > 0 else np.nan,
                     "hotel_pearson": pearsonr(ph, hotel_mean).statistic if ph.std() > 0 else np.nan,
                     "hotel_spearman": spearmanr(ph, hotel_mean).statistic if ph.std() > 0 else np.nan})
        if pr.std() > 0:
            row = {"model": name}
            for t, g in kr.groupby("traveler_type"):
                row[t] = spearmanr(p[g.hr_id], g.rating).statistic
            type_rows.append(row)
    res = pd.DataFrame(rows)
    print(f"Mean score: Kaggle {kr.rating.mean():.3f} / HotelRec training μ {float(s['base_mu']):.3f}")
    print(res.round(4).to_string(index=False))
    save_table(res, "11_kaggle_matched_prediction")
    by_type = pd.DataFrame(type_rows)
    print(by_type.round(4).to_string(index=False))
    save_table(by_type, "11_kaggle_matched_by_traveler_type")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, name in zip(ax, ["MF(μ+b_i)", f"Hybrid-rating (MF+Content, λ={lam})"]):
        ph = preds[name][hotel_mean.index]
        a.scatter(ph, hotel_mean, s=8, alpha=0.5)
        a.set_xlabel(f"HotelRec prediction ({name})")
        a.set_ylabel("Kaggle hotel mean score (5-point)")
        a.set_title(f"r = {pearsonr(ph, hotel_mean).statistic:.3f}")
    plt.suptitle(f"{len(hotel_mean):,} matched hotels: HotelRec new-user prediction vs actual Booking.com")
    save_fig(plt, "11_kaggle_matched_prediction")


if __name__ == "__main__":
    main()
