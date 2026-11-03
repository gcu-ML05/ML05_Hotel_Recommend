"""
Step 8: generalization check on the Kaggle (Booking.com) data

The Kaggle data has no user IDs and no sub-ratings, so, as in section 2 of the proposal, each review is treated
as one new user and their preferences are estimated from the aspects mentioned in the review text.

(1) Aspect mention extraction
    Using a per-aspect keyword lexicon (LEXICON, a hand-made list chosen by the team), count whether each aspect
    is mentioned in the positive/negative review text.
    Aspect mention share of a review = mentions / total mentions  (reviews with no mention are dropped)

(2) Aspect mentions by trip type
    How much the mention shares per traveler_type (Couple, Business, ...) and trip_type differ from the
    overall average (percentage points)

(3) Correspondence with the HotelRec personas
    Assign each review the persona with the highest cosine similarity between the review's mention-share deviation
    (from the overall average) and the persona centroid's deviation (from the HotelRec overall average), then test
    the independence of the traveler type x persona contingency table with a chi-square test.

(4) Generalization of a content-based score
    Hotel aspect reputation = mean of (positive mention - negative mention) over the hotel's other reviews
    (own review excluded, leave-one-out)
    How well hotel score = Σ_k weight_k x reputation_k correlates with the reviewer's actual score (Spearman rank correlation)
    Weights: equal / HotelRec global weights / weights of the persona assigned to the review

Output: Pipeline/outputs/08_*.csv, figures/08_*.png
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from common import ASPECTS, KAGGLE_DIR, WORK_DIR, ensure_dirs, save_fig, save_table, setup_plot

# keywords per aspect (lower case, matched on word boundaries). "bed" counts as rooms, mattress/pillow as sleep_quality
LEXICON = {
    "service": ["staff", "service", "reception", "receptionist", "concierge", "friendly", "helpful",
                "rude", "polite", "welcoming", "check in", "checkin"],
    "cleanliness": ["clean", "cleanliness", "dirty", "dust", "dusty", "stain", "stained", "smell", "smelly",
                    "hygiene", "spotless", "mould", "mold"],
    "value": ["price", "prices", "value", "money", "expensive", "cheap", "overpriced", "worth", "cost",
              "pricey", "charged"],
    "rooms": ["room", "rooms", "bathroom", "shower", "bed", "beds", "furniture", "spacious", "tiny",
              "decor", "balcony", "towels"],
    "location": ["location", "located", "metro", "station", "tube", "walk", "walking", "distance", "central",
                 "centre", "center", "transport", "airport", "nearby", "neighbourhood"],
    "sleep_quality": ["sleep", "slept", "noise", "noisy", "quiet", "loud", "mattress", "pillow", "pillows",
                      "soundproof", "insulation"],
}
PATTERNS = {a: r"\b(?:" + "|".join(re.escape(w) for w in ws) + r")\b" for a, ws in LEXICON.items()}


def main():
    ensure_dirs()
    plt = setup_plot()
    k = pd.read_csv(KAGGLE_DIR / "kaggle_test.csv", encoding="utf-8-sig",
                    usecols=["review_id", "hotel_id", "rating", "trip_type", "traveler_type",
                             "positive_review", "negative_review"])
    print(f"Kaggle reviews: {len(k):,}")

    # (1) aspect mentions (positive/negative separately: mentioned 0/1)
    pos_txt = k.positive_review.fillna("").str.lower()
    neg_txt = k.negative_review.fillna("").str.lower()
    POS = np.column_stack([pos_txt.str.contains(PATTERNS[a], regex=True) for a in ASPECTS]).astype(float)
    NEG = np.column_stack([neg_txt.str.contains(PATTERNS[a], regex=True) for a in ASPECTS]).astype(float)
    M = POS + NEG
    has = M.sum(1) > 0
    share = M[has] / M[has].sum(1, keepdims=True)
    kk = k[has].reset_index(drop=True)
    print(f"reviews with an aspect mention: {has.sum():,} ({has.mean():.1%})")
    overall = share.mean(0)
    print("overall mention share:", dict(zip(ASPECTS, overall.round(3))))

    # (2) mention share by trip type (percentage points vs the overall average)
    S = pd.DataFrame(share, columns=ASPECTS)
    tabs = []
    for col in ["traveler_type", "trip_type"]:
        g = S.groupby(kk[col]).mean()
        dev = (g - overall) * 100
        dev.insert(0, "n_reviews", kk[col].value_counts().reindex(dev.index))
        tabs.append(dev.rename_axis("group").reset_index().assign(column=col))
    dev_tab = pd.concat(tabs, ignore_index=True)
    print(dev_tab.round(2).to_string(index=False))
    save_table(dev_tab.round(3), "08_aspect_mention_by_trip_type")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    D = dev_tab.set_index("group")[ASPECTS]
    lim = np.abs(D.to_numpy()).max()
    im = ax.imshow(D.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(ASPECTS)), ASPECTS, rotation=30)
    ax.set_yticks(range(len(D)), D.index)
    for a in range(D.shape[0]):
        for b in range(D.shape[1]):
            ax.text(b, a, f"{D.iat[a, b]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Aspect mention share by trip type minus overall average (pp)")
    plt.colorbar(im)
    save_fig(plt, "08_aspect_mention_by_trip_type")

    # (3) correspondence with personas
    cent = pd.read_csv(WORK_DIR / "persona_centroids.csv")
    users = pd.read_csv(WORK_DIR / "user_personas.csv")
    base = users[users.n_full >= 3][ASPECTS].mean().to_numpy()       # mean share of the users used for persona fitting
    C = cent[ASPECTS].to_numpy() - base
    R = share - overall

    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    kk["persona"] = cent["name"].to_numpy()[(unit(R) @ unit(C).T).argmax(1)]
    ct = pd.crosstab(kk.traveler_type, kk.persona)
    # chi2_contingency(table): tests whether two categorical variables are independent. Returns (chi-square, p-value, dof, expected counts)
    chi2, p, dof, _ = chi2_contingency(ct)
    cramer_v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))   # 0 to 1, strength of association
    print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1).to_string())
    print(f"chi-square {chi2:.1f}, dof {dof}, p={p:.3g}, Cramér's V = {cramer_v:.3f}")
    save_table((ct.div(ct.sum(axis=1), axis=0) * 100).round(2), "08_traveler_type_x_persona", index=True)

    # which persona the mention profile of each traveler type resembles most
    G = S.groupby(kk.traveler_type).mean().to_numpy() - overall
    sim = pd.DataFrame(unit(G) @ unit(C).T, index=S.groupby(kk.traveler_type).mean().index, columns=cent["name"])
    print(sim.round(3).to_string())
    save_table(sim.round(4), "08_traveler_type_persona_similarity", index=True)

    # (4) generalization of a content score: rank correlation between hotel aspect reputation (leave-one-out) and the reviewer's score
    s_r = POS - NEG                                                   # per-review aspect reputation (-1, 0, 1)
    hid = k.hotel_id.to_numpy()
    n_h = np.bincount(hid)
    S_h = np.column_stack([np.bincount(hid, s_r[:, j]) for j in range(len(ASPECTS))])
    loo = (S_h[hid] - s_r) / np.maximum(n_h[hid] - 1, 1)[:, None]    # hotel reputation without the review itself
    loo = loo[has]
    w_g = np.load(WORK_DIR / "global_weights.npy")[1:]
    w_p = cent.set_index("name").loc[kk.persona, ASPECTS].to_numpy()
    scores = {"equal weights": loo.mean(1), "HotelRec global weights": loo @ (w_g / w_g.sum()),
              "assigned persona weights": (loo * w_p).sum(1)}
    rows = []
    for name, sc in scores.items():
        row = {"weights": name, "all": spearmanr(sc, kk.rating).statistic}   # Spearman rank correlation
        for t, g in kk.groupby("traveler_type"):
            row[t] = spearmanr(sc[g.index], g.rating).statistic
        rows.append(row)
    corr = pd.DataFrame(rows)
    print(corr.round(4).to_string(index=False))
    save_table(corr.round(4), "08_content_score_generalization")


if __name__ == "__main__":
    main()
