"""
Step 10: plug the new personas and the revised keyword lexicon into the Kaggle generalization check of 08 and compare

08_kaggle_generalization.py uses (a) the original personas from 02 and (b) the hand-made LEXICON.
Here the following four things are compared on the same metrics.

(1) Personas: original (02, regression-based) vs deviation-based (04)
(2) Assignment text: all positive+negative mentions vs negative mentions only (complaints in negative reviews map more directly to 'strictness')
(3) Keyword lexicon: original LEXICON vs revised LEXICON (reflecting the 09 audit)
(4) Lexicon robustness: whether the conclusions on aspect mention deviation by trip type hold when the lexicon changes

Metrics: Cramér's V (strength of the traveler type x persona association), Spearman correlation (content score vs reviewer score)

Prerequisites: 01~05 (02's persona_centroids.csv, the deviation-based personas from 04·05), 08 (LEXICON)
Output: Pipeline/outputs/10_*.csv
"""
import importlib

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from common import ASPECTS, KAGGLE_DIR, OUT_DIR, WORK_DIR, ensure_dirs, save_table

# the file name starts with a digit, so an import statement cannot be used; load it with importlib.import_module(module name) (reuse the original LEXICON)
base_mod = importlib.import_module("08_kaggle_generalization")
LEXICON_ORIG = base_mod.LEXICON

# revised lexicon: remove the ambiguous keywords flagged in the 09 audit + add missing keywords
REMOVE = {
    "rooms": ["bed", "beds"],          # "bed was comfortable" overlaps with sleep_quality
    "cleanliness": ["smell"],          # cannot tell "smell of smoke" from "smells good breakfast" (keep only smelly from the original)
    "value": ["cheap"],                # "cheap furniture" is a complaint about rooms quality
}
ADD = {
    "service": ["waiter", "waitress", "manager", "housekeeping"],
    "cleanliness": ["hair", "filthy", "unclean"],
    "value": ["overcharged", "affordable", "reasonable", "bargain"],
    "location": ["restaurants", "shops", "attractions", "close to"],
    "sleep_quality": ["comfortable bed", "uncomfortable", "thin walls", "traffic", "bedding", "snoring"],
}


def build_patterns(lexicon):
    """Keyword lexicon -> regex per aspect (word-boundary matching). Same as 08."""
    import re
    return {a: r"\b(?:" + "|".join(re.escape(w) for w in ws) + r")\b" for a, ws in lexicon.items()}


def revised_lexicon():
    """Revised lexicon: the original minus REMOVE plus ADD (duplicates removed, order kept)."""
    lex = {}
    for a, ws in LEXICON_ORIG.items():
        kept = [w for w in ws if w not in REMOVE.get(a, [])]
        lex[a] = list(dict.fromkeys(kept + ADD.get(a, [])))
    return lex


def mentions(pos_txt, neg_txt, lexicon):
    """Per-review aspect mention matrices (positive POS, negative NEG; 0/1, reviews x 6 aspects)."""
    pat = build_patterns(lexicon)
    # Series.str.contains(regex): boolean for whether the regex occurs in the text
    pos = np.column_stack([pos_txt.str.contains(pat[a], regex=True) for a in ASPECTS]).astype(float)
    neg = np.column_stack([neg_txt.str.contains(pat[a], regex=True) for a in ASPECTS]).astype(float)
    return pos, neg


def unit(x):
    """Row-wise normalization (for cosine similarity)."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_personas():
    """Load the persona sets: (centroid table, baseline share). The baseline is the 'average profile' subtracted before the cosine comparison."""
    old = pd.read_csv(WORK_DIR / "persona_centroids.csv")
    users = pd.read_csv(WORK_DIR / "user_personas.csv")
    base_old = users[users.n_full >= 3][ASPECTS].mean().to_numpy()
    new = pd.read_csv(OUT_DIR / "04_persona_dev_share.csv")
    cnt = pd.read_csv(OUT_DIR / "04_persona_dev_centroids.csv").set_index("name").loc[new["name"], "n_users_fit"]
    # baseline of the new personas = centroid mean weighted by the number of training users
    base_new = (new[ASPECTS].to_numpy() * cnt.to_numpy()[:, None]).sum(0) / cnt.sum()
    out = {"regression-based (02)": (old, base_old), "deviation-based (04)": (new, base_new)}
    # if the 05 results (improved shrinkage strength and K) exist, compare them too
    tuned_path = OUT_DIR / "05_persona_dev_share.csv"
    if tuned_path.exists():
        tuned = pd.read_csv(tuned_path)
        w = tuned["n_users_fit"].to_numpy()[:, None]
        out["deviation-based improved (05)"] = (tuned, (tuned[ASPECTS].to_numpy() * w).sum(0) / w.sum())
    return out


def evaluate(POS, NEG, meta, cent, base, mode, w_g):
    """Metrics for one combination (persona set x assignment text).

    Assignment: the persona with the highest cosine similarity between the review's mention-share deviation R
    and the centroid deviation C = centroid - baseline.
    Content score: hotel aspect reputation (leave-one-out) multiplied by the weights and summed; Spearman correlation with the reviewer score.
    """
    M = (POS + NEG) if mode == "all text" else NEG
    has = M.sum(1) > 0
    share = M[has] / M[has].sum(1, keepdims=True)
    R = share - share.mean(0)
    C = cent[ASPECTS].to_numpy() - base
    pidx = (unit(R) @ unit(C).T).argmax(1)
    trav = meta["traveler_type"].to_numpy()[has]
    ct = pd.crosstab(trav, cent["name"].to_numpy()[pidx])
    chi2, p, _, _ = chi2_contingency(ct)       # chi-square test of independence: (statistic, p-value, dof, expected counts)
    v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))

    s_r = POS - NEG                             # per-review aspect reputation (-1, 0, 1)
    hid = meta["hotel_id"].to_numpy()
    n_h = np.bincount(hid)                      # np.bincount: number of reviews per hotel
    S_h = np.column_stack([np.bincount(hid, s_r[:, j]) for j in range(len(ASPECTS))])
    loo = ((S_h[hid] - s_r) / np.maximum(n_h[hid] - 1, 1)[:, None])[has]   # hotel reputation excluding the review itself
    y = meta["rating"].to_numpy()[has]
    w_p = cent[ASPECTS].to_numpy()[pidx]
    # spearmanr(x, y): rank correlation coefficient (strength of a monotonic relation, -1~1). .statistic is the coefficient
    rho = {"equal weights": spearmanr(loo.mean(1), y).statistic,
           "global weights": spearmanr(loo @ (w_g / w_g.sum()), y).statistic,
           "persona weights": spearmanr((loo * w_p).sum(1), y).statistic}
    sizes = pd.Series(cent["name"].to_numpy()[pidx]).value_counts(normalize=True)
    return {"n_reviews": int(has.sum()), "chi2": chi2, "p": p, "cramers_v": v,
            "rho_equal": rho["equal weights"], "rho_global": rho["global weights"], "rho_persona": rho["persona weights"],
            "largest_persona_share": sizes.max()}


def trip_deviation(POS, NEG, meta):
    """Aspect mention share by trip type - overall average (pp). Same definition as 08, rows = traveler_type/trip_type groups."""
    M = POS + NEG
    has = M.sum(1) > 0
    S = pd.DataFrame(M[has] / M[has].sum(1, keepdims=True), columns=ASPECTS)
    overall = S.mean().to_numpy()
    out = []
    for col in ["traveler_type", "trip_type"]:
        g = S.groupby(meta[col].to_numpy()[has]).mean()
        out.append(((g - overall) * 100).assign(column=col))
    return pd.concat(out)


def main():
    """Evaluate the 2 lexicons x persona sets x assignment text combinations, and save the lexicon robustness and change list."""
    ensure_dirs()
    k = pd.read_csv(KAGGLE_DIR / "kaggle_test.csv", encoding="utf-8-sig",
                    usecols=["hotel_id", "rating", "trip_type", "traveler_type", "positive_review", "negative_review"])
    pos_txt = k.positive_review.fillna("").str.lower()
    neg_txt = k.negative_review.fillna("").str.lower()
    w_g = np.load(WORK_DIR / "global_weights.npy")[1:]
    personas = load_personas()
    lexicons = {"original lexicon": LEXICON_ORIG, "revised lexicon": revised_lexicon()}

    rows, devs = [], {}
    for lname, lex in lexicons.items():
        POS, NEG = mentions(pos_txt, neg_txt, lex)
        devs[lname] = trip_deviation(POS, NEG, k)
        for pname, (cent, base) in personas.items():
            for mode in ["all text", "negative text only"]:
                rows.append({"lexicon": lname, "persona": pname, "assign_by": mode,
                             **evaluate(POS, NEG, k, cent, base, mode, w_g)})
    comp = pd.DataFrame(rows)
    print(comp.round(4).to_string(index=False))
    save_table(comp.round(4), "10_persona_comparison")

    # (4) lexicon robustness: whether the per-group deviation table stays the same after the lexicon revision
    a, b = devs["original lexicon"], devs["revised lexicon"]
    A, B = a[ASPECTS], b[ASPECTS]
    same_top = (A.idxmax(axis=1) == B.idxmax(axis=1))
    same_bottom = (A.idxmin(axis=1) == B.idxmin(axis=1))
    rob = pd.DataFrame({"column": a["column"], "top_orig": A.idxmax(axis=1), "top_rev": B.idxmax(axis=1),
                        "top_same": same_top, "bottom_orig": A.idxmin(axis=1), "bottom_rev": B.idxmin(axis=1),
                        "bottom_same": same_bottom, "max_abs_diff_pp": (A - B).abs().max(axis=1)})
    print(rob.round(2).to_string())
    flat = np.corrcoef(A.to_numpy().ravel(), B.to_numpy().ravel())[0, 1]
    print(f"Correlation of the group x aspect deviation tables (original vs revised): {flat:.3f}, top aspect kept {same_top.mean():.0%}, "
          f"bottom kept {same_bottom.mean():.0%}")
    save_table(rob.round(3), "10_lexicon_robustness", index=True)
    save_table(B.assign(column=b["column"]).round(3), "10_aspect_mention_by_trip_type_revised", index=True)

    # record the revised lexicon changes (for team review)
    changes = [{"aspect": x, "action": "remove", "keyword": w} for x, ws in REMOVE.items() for w in ws]
    changes += [{"aspect": x, "action": "add", "keyword": w} for x, ws in ADD.items() for w in ws]
    save_table(pd.DataFrame(changes), "10_lexicon_changes")


if __name__ == "__main__":
    main()
