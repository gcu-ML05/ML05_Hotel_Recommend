"""
Validation of the preprocessing results (HotelRec for training + Kaggle for testing)

Automatically checks that the preprocessing scripts behaved as intended and prints a PASS/FAIL list and summary statistics.
The results are also saved as validation_report.csv in each reports folder.

Usage
    python 03_validate_preprocessing.py --hotelrec-dir ../data/processing/hotelrec --kaggle-dir ../data/processing/kaggle
    (If you changed --min-user/--min-hotel in the HotelRec preprocessing, pass the same values.
     If --max-users sampling was used, pass the post-sampling threshold (default 10) as --min-hotel.)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ASPECTS = ["service", "cleanliness", "value", "rooms", "location", "sleep_quality"]
PLACEHOLDER_AUTHORS = {"/undefined", "undefined", "null", "none", "nan", "",
                       "a tripadvisor member", "lonely planet traveler"}
TRAVELER_TYPES = {"Couple", "Solo traveler", "Family with young children",
                  "Family with older children", "Group", "Travelers with friends"}
TRIP_TYPES = {"Leisure trip", "Business trip"}

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def validate_hotelrec(d, min_user, min_hotel):
    d = Path(d)
    print(f"\n=== HotelRec validation: {d} ===")
    files = ["hotelrec_train.csv", "hotelrec_partial.csv", "user_map.csv", "hotel_map.csv"]
    missing = [f for f in files if not (d / f).exists()]
    check("required files exist", not missing, f"missing files: {missing}" if missing else "")
    if missing:
        return

    cols = ["user_id", "hotel_id", "rating", "date", *ASPECTS, "title", "text"]
    t = pd.read_csv(d / "hotelrec_train.csv")
    p = pd.read_csv(d / "hotelrec_partial.csv")
    um = pd.read_csv(d / "user_map.csv")
    hm = pd.read_csv(d / "hotel_map.csv")
    check("column layout (train/partial)", list(t.columns) == cols and list(p.columns) == cols)

    a = pd.concat([t, p], ignore_index=True)
    print(f"  train {len(t):,} rows / partial {len(p):,} rows / total {len(a):,} rows")

    # C. split criterion
    check("train: no missing values in the 6 sub-ratings", t[ASPECTS].notna().all().all(),
          f"{int(t[ASPECTS].isna().sum().sum())} missing")
    check("partial: every row has at least 1 missing sub-rating", p[ASPECTS].isna().any(axis=1).all(),
          f"{int(p[ASPECTS].notna().all(axis=1).sum())} complete rows")

    # D. duplicates
    check("no duplicate (user, hotel) pairs", not a.duplicated(["user_id", "hotel_id"]).any(),
          f"{int(a.duplicated(['user_id', 'hotel_id']).sum())} duplicates")

    # value ranges / missing values
    check("rating in 1~5, no missing values", a.rating.between(1, 5).all(), f"{int(a.rating.isna().sum())} missing")
    asp_ok = all(a[c].dropna().between(1, 5).all() for c in ASPECTS)
    check("sub-rating values in the 1~5 range", asp_ok)
    dt = pd.to_datetime(a.date, errors="coerce")
    check("date can be converted", dt.notna().all(), f"range {dt.min().date()} ~ {dt.max().date()}")

    # B. minimum review counts
    uc, hc = a.user_id.value_counts(), a.hotel_id.value_counts()
    check(f"minimum reviews per user >= {min_user}", uc.min() >= min_user, f"min {uc.min()}")
    check(f"minimum reviews per hotel >= {min_hotel}", hc.min() >= min_hotel, f"min {hc.min()}")

    # ID consistency
    check("every user_id exists in user_map", set(a.user_id) <= set(um.user_id))
    check("every hotel_id exists in hotel_map", set(a.hotel_id) <= set(hm.hotel_id))
    check("user_id/hotel_id unique (map files)", um.user_id.is_unique and hm.hotel_id.is_unique)
    ph = um.author.fillna("").astype(str).str.strip().str.lower().isin(PLACEHOLDER_AUTHORS)
    check("no anonymous authors ('/undefined' etc.)", not ph.any(), f"{int(ph.sum())} users")

    # text
    et = int((a.text.fillna("").str.strip() == "").sum())
    print(f"  (note) reviews with empty text: {et:,} ({et / len(a):.2%})")

    # reference metric important for modeling: distribution of reviews per train user
    c = t.user_id.value_counts()
    print("\n  [note] distribution of reviews per user in train (complete sub-ratings)")
    for n in [1, 3, 5, 7, 10]:
        print(f"    {n} or more: {(c >= n).sum():>8,} users ({(c >= n).mean():6.1%})")
    print(f"    mean {c.mean():.2f}, median {c.median():.0f}, {len(c):,} users")
    sp = 1 - len(a) / (a.user_id.nunique() * a.hotel_id.nunique())
    print(f"  [note] overall (train+partial) sparsity {sp:.4%}")


def validate_kaggle(d):
    d = Path(d)
    print(f"\n=== Kaggle validation: {d} ===")
    f, mf = d / "kaggle_test.csv", d / "hotel_meta_kaggle.csv"
    missing = [x.name for x in (f, mf) if not x.exists()]
    check("required files exist", not missing, f"missing files: {missing}" if missing else "")
    if missing:
        return
    k = pd.read_csv(f, keep_default_na=False, na_values=[""])
    meta = pd.read_csv(mf)
    print(f"  {len(k):,} rows, {k.hotel_id.nunique():,} hotels")

    check("review_id unique", k.review_id.is_unique)
    check("rating in the 1~5 range, no missing values", k.rating.between(1, 5).all(), f"{int(k.rating.isna().sum())} missing")
    check("rating = reviewer_score_10 / 2",
          np.allclose(k.rating, (k.reviewer_score_10 / 2).round(2), atol=0.011))
    check("review_date can be converted", pd.to_datetime(k.review_date, errors="coerce").notna().all(),
          f"{k.review_date.min()} ~ {k.review_date.max()}")
    # an error only when the whole field is the placeholder phrase (the same words appearing inside a text are fine)
    check("no placeholder phrases (positive/negative fields are not 'No Positive'/'No Negative')",
          not (k.positive_review.eq("No Positive").any() or k.negative_review.eq("No Negative").any()))
    check("traveler_type values in the allowed list", k.traveler_type.dropna().isin(TRAVELER_TYPES).all())
    check("trip_type values in the allowed list", k.trip_type.dropna().isin(TRIP_TYPES).all(),
          f"{int(k.trip_type.isna().sum()):,} missing")
    check("every hotel_id exists in hotel_meta", set(k.hotel_id) <= set(meta.hotel_id))
    # short reviews ("Just perfect" etc.) can be written by different people on the same day with the same score, so only texts of 40+ characters are checked
    body = k.drop(columns="review_id")
    long_dup = int(body[body.review_text.fillna("").str.len() >= 40].duplicated().sum())
    check("no duplicate rows (texts of 40+ characters)", long_dup == 0, f"{long_dup} duplicates")
    print(f"  (note) duplicates overall including short texts: {int(body.duplicated().sum())}")
    print(f"  (note) reviews with empty text: {int(k.review_text.isna().sum()):,}")
    print(f"  (note) traveler type distribution: {k.traveler_type.value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotelrec-dir", default="../data/processing/hotelrec")
    ap.add_argument("--kaggle-dir", default="../data/processing/kaggle")
    ap.add_argument("--min-user", type=int, default=5)
    ap.add_argument("--min-hotel", type=int, default=10)
    args = ap.parse_args()

    validate_hotelrec(args.hotelrec_dir, args.min_user, args.min_hotel)
    validate_kaggle(args.kaggle_dir)

    res = pd.DataFrame(results)
    n_fail = int((res.result == "FAIL").sum())
    print(f"\n==== Validation summary: PASS {len(res) - n_fail}, FAIL {n_fail} of {len(res)} ====")
    for d in (args.hotelrec_dir, args.kaggle_dir):
        rp = Path(d) / "reports"
        if rp.exists():
            res.to_csv(rp / "validation_report.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
