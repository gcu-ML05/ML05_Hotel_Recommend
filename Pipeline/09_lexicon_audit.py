"""
Step 9: audit of the Kaggle aspect keyword lexicon (LEXICON)

The results of 08_kaggle_generalization.py (aspect mention shares by trip type, persona correspondence) depend on
a hand-made keyword list. This script builds the evidence before the lexicon is revised.

(1) Frequency per keyword: share of positive/negative reviews containing each keyword. If one word accounts for
    most of an aspect's mentions (e.g. "room" for rooms), the result is driven by that single word.
(2) Co-mention between aspects: how often two aspects are caught in the same review. Finds aspects whose
    boundaries overlap, such as "bed" (rooms) and "mattress" (sleep_quality).
(3) Sensitivity: how much the mention deviation by trip type (pp) changes when keywords are removed one at a time.
    Keywords that change it a lot drive the result, so they get priority in the team review.

Run from: copy into the team repo's Pipeline/ folder, then  cd Pipeline && python 09_lexicon_audit.py
Output: Pipeline/outputs/09_keyword_freq.csv, 09_aspect_cooccurrence.csv, 09_keyword_sensitivity.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common import ASPECTS, KAGGLE_DIR, SEED, ensure_dirs, save_table

# reuse LEXICON from 08_kaggle_generalization.py as is (so the lexicon is maintained in one place only)
# the file name starts with a digit, so a normal import does not work; read it with importlib
# spec_from_file_location(module name, file path): build a module spec from a file / module_from_spec: create an empty module
# spec.loader.exec_module(module): run the file's code to fill the module (08's main() does not run thanks to the __name__ check)
_spec = importlib.util.spec_from_file_location("kaggle08", Path(__file__).parent / "08_kaggle_generalization.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LEXICON = _mod.LEXICON

SAMPLE = 100_000      # sensitivity is computed on a sample (number of keywords x recomputation, so the full data is slow)


def contains(texts, words):
    """Whether each review in texts contains any of words (matched on word boundaries)"""
    import re
    # re.escape(string): escape regex special characters, \b: word boundary (so room is not caught inside bathroom)
    pat = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
    # Series.str.contains(regex, regex=True): boolean for whether each text contains the regex
    return texts.str.contains(pat, regex=True).to_numpy()


def trip_type_dev(pos, neg, lex, groups):
    """Deviation (pp) of the aspect mention share by trip type from the overall average, computed as in 08"""
    M = np.column_stack([contains(pos, lex[a]) for a in ASPECTS]).astype(float) \
        + np.column_stack([contains(neg, lex[a]) for a in ASPECTS]).astype(float)
    has = M.sum(1) > 0
    share = M[has] / M[has].sum(1, keepdims=True)
    S = pd.DataFrame(share, columns=ASPECTS)
    g = S.groupby(groups[has].to_numpy()).mean()
    return (g - share.mean(0)) * 100


def main():
    """Compute (1) keyword frequency -> (2) aspect co-mention -> (3) keyword removal sensitivity and save the tables."""
    ensure_dirs()
    k = pd.read_csv(KAGGLE_DIR / "kaggle_test.csv", encoding="utf-8-sig",
                    usecols=["traveler_type", "positive_review", "negative_review"])
    pos = k.positive_review.fillna("").str.lower()
    neg = k.negative_review.fillna("").str.lower()

    # (1) frequency per keyword
    rows = []
    for a, words in LEXICON.items():
        hit_a = contains(pos, words) | contains(neg, words)
        for w in words:
            hp, hn = contains(pos, [w]), contains(neg, [w])
            rows.append({"aspect": a, "keyword": w, "pos_%": 100 * hp.mean(), "neg_%": 100 * hn.mean(),
                         # share of reviews where this aspect was caught that contain this keyword (high = aspect depends on this word)
                         "share_of_aspect_hits_%": 100 * (hp | hn)[hit_a].mean() if hit_a.any() else 0})
    freq = pd.DataFrame(rows).sort_values(["aspect", "share_of_aspect_hits_%"], ascending=[True, False])
    print(freq.round(2).to_string(index=False))
    save_table(freq.round(3), "09_keyword_freq")

    # (2) co-mention between aspects (Jaccard similarity)
    H = np.column_stack([contains(pos, LEXICON[a]) | contains(neg, LEXICON[a]) for a in ASPECTS])
    J = np.zeros((len(ASPECTS), len(ASPECTS)))
    for i in range(len(ASPECTS)):
        for j in range(len(ASPECTS)):
            J[i, j] = (H[:, i] & H[:, j]).sum() / max((H[:, i] | H[:, j]).sum(), 1)
    co = pd.DataFrame(J, index=ASPECTS, columns=ASPECTS)
    print(co.round(3).to_string())
    save_table(co.round(4), "09_aspect_cooccurrence", index=True)

    # (3) sensitivity: largest change (pp) in the deviation by trip type when one keyword is removed
    s = k.sample(n=min(SAMPLE, len(k)), random_state=SEED).reset_index(drop=True)
    sp, sn, grp = s.positive_review.fillna("").str.lower(), s.negative_review.fillna("").str.lower(), s.traveler_type
    base = trip_type_dev(sp, sn, LEXICON, grp)
    rows = []
    for a, words in LEXICON.items():
        if len(words) == 1:
            continue
        for w in words:
            lex = {**LEXICON, a: [x for x in words if x != w]}
            d = trip_type_dev(sp, sn, lex, grp)
            rows.append({"aspect": a, "removed_keyword": w,
                         "max_abs_change_pp": float(np.abs(d - base).to_numpy().max())})
    sens = pd.DataFrame(rows).sort_values("max_abs_change_pp", ascending=False)
    print(sens.head(20).round(3).to_string(index=False))
    save_table(sens.round(4), "09_keyword_sensitivity")


if __name__ == "__main__":
    main()
