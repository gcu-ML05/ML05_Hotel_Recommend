"""
HotelRec (training data) preprocessing

Input : HotelRec.txt  (JSON Lines, one JSON per line, about 50 million reviews)
Output : data/processing/hotelrec/
    hotelrec_train.csv      reviews with all 6 sub-ratings (for model training)
    hotelrec_partial.csv    reviews missing some/all sub-ratings (saved separately instead of dropped)
    hotel_map.csv           hotel_id <-> hotel_url / hotel name / location
    user_map.csv            user_id <-> author
    reports/hotelrec_report.csv   before/after comparison per step (rows/users/hotels, missing values, sparsity)

Team decisions
    A. No region restriction (worldwide)
    B. Users with at least MIN_USER reviews, hotels with at least MIN_HOTEL reviews (iterative filtering, k-core)
    C. Only reviews with all 6 sub-ratings go to hotelrec_train.csv, the rest are saved separately to hotelrec_partial.csv
    D. For duplicate (user, hotel) pairs keep only the most recent review
    (E is applied in the Kaggle preprocessing)

Processing order (the original is read one line at a time to save memory)
    1) First scan : read the whole original and count reviews per user/hotel
    2) Second scan : save only candidate rows (users >= MIN_USER, hotels >= MIN_HOTEL reviews) to an interim file
    3) In-memory processing : format validation -> duplicate removal (latest) -> iterative filtering -> ID assignment -> split save

Usage
    python 01_preprocess_hotelrec.py --src ../data/raw/HotelRec.txt
"""
import argparse
import csv
import json
import re
import shutil
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- default settings (can be changed with options) ----------------
MIN_USER = 5      # minimum reviews per user
MIN_HOTEL = 20    # minimum reviews per hotel
# ---------------------------------------------------------------

ASPECTS = {  # property_dict key -> column name (the 6 aspects of the proposal)
    "service": "service", "cleanliness": "cleanliness", "value": "value",
    "rooms": "rooms", "location": "location", "sleep quality": "sleep_quality",
}
ASPECT_COLS = list(ASPECTS.values())
# anonymous/placeholder authors that are not real users (removed, since grouping them as one user distorts training)
PLACEHOLDER_AUTHORS = {"/undefined", "undefined", "null", "none", "nan", "",
                       "a tripadvisor member", "lonely planet traveler"}
URL_RE = re.compile(r"-d(?P<d>\d+)-Reviews-(?P<name>.+)-(?P<loc>[^-]+)\.html$")


def iter_records(path):
    """Read JSON Lines one line at a time and yield dicts (broken lines are skipped)"""
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    iter_records.bad = bad


def is_real_author(a):
    return isinstance(a, str) and a.strip().lower() not in PLACEHOLDER_AUTHORS


def stats(df, stage, user="author", hotel="hotel_url"):
    """One row of summary statistics per step"""
    n = len(df)
    nu, nh = df[user].nunique(), df[hotel].nunique()
    row = {"stage": stage, "rows": n, "users": nu, "hotels": nh,
           "density_%": round(100 * n / (nu * nh), 6) if nu and nh else 0,
           "avg_reviews_per_user": round(n / nu, 2) if nu else 0}
    for c in ["rating"] + ASPECT_COLS:
        if c in df:
            row[f"missing_{c}_%"] = round(100 * df[c].isna().mean(), 2)
    return row


def kcore(df, min_user, min_hotel):
    """Filter repeatedly until the minimum review counts per user/hotel are met"""
    it = 0
    while True:
        it += 1
        n0 = len(df)
        uc = df.author.value_counts()
        df = df[df.author.isin(uc[uc >= min_user].index)]
        hc = df.hotel_url.value_counts()
        df = df[df.hotel_url.isin(hc[hc >= min_hotel].index)]
        if len(df) == n0:
            return df, it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/raw/HotelRec.txt")
    ap.add_argument("--out-dir", default="../data/processing/hotelrec")
    ap.add_argument("--min-user", type=int, default=MIN_USER)
    ap.add_argument("--min-hotel", type=int, default=MIN_HOTEL)
    ap.add_argument("--max-users", type=int, default=200_000,
                    help="maximum number of users kept at random after iterative filtering (0 = no sampling)")
    ap.add_argument("--min-hotel-after-sampling", type=int, default=10,
                    help="minimum reviews per hotel for the refiltering after sampling (fewer users means fewer reviews per hotel, so the threshold is relaxed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="skip the first/second scans and use the existing interim file and reports/raw_stats.json")
    ap.add_argument("--interim", default=None, help="interim file path (default: out-dir/interim_candidates.csv)")
    # when raw_stats.json is missing (the first/second scans were run with an older version), enter the original statistics printed in the log
    ap.add_argument("--raw-total", type=int, help="total lines in the original (e.g. 50264531)")
    ap.add_argument("--raw-placeholder", type=int, help="lines from anonymous authors that were excluded (e.g. 1216923)")
    ap.add_argument("--raw-users", type=int, help="number of users after excluding anonymous authors (e.g. 21891400)")
    ap.add_argument("--raw-hotels", type=int, help="number of hotels (e.g. 365055)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    interim = Path(args.interim) if args.interim else out / "interim_candidates.csv"
    raw_stats_path = out / "reports" / "raw_stats.json"
    report = []
    print(f"Output location: {out.resolve()}  (free disk space {shutil.disk_usage(out).free / 1e9:.1f}GB)")

    if args.resume:
        print("[1~2/3] --resume: skipping the scans and using the existing interim file")
        if raw_stats_path.exists():
            report = json.load(open(raw_stats_path, encoding="utf-8"))
        elif None not in (args.raw_total, args.raw_placeholder, args.raw_users, args.raw_hotels):
            tot, ph, nu, nh = args.raw_total, args.raw_placeholder, args.raw_users, args.raw_hotels
            report = [
                {"stage": "0a_raw (full original)", "rows": tot,
                 "users": nu + (1 if ph else 0), "hotels": nh},
                {"stage": "0b_remove reviews by anonymous authors ('/undefined', 'null', etc.)", "rows": tot - ph,
                 "users": nu, "hotels": nh,
                 "density_%": round(100 * (tot - ph) / (nu * nh), 8),
                 "avg_reviews_per_user": round((tot - ph) / nu, 2)},
            ]
        else:
            raise SystemExit("raw_stats.json not found. Pass --raw-total, --raw-placeholder, "
                             "--raw-users and --raw-hotels together (the numbers printed in the first scan log).")
        if not interim.exists():
            raise SystemExit(f"Interim file not found: {interim}  (set the path with --interim)")
    else:
        # ------------------------------------------------------------
        # First scan: count reviews per user/hotel
        #   store 64-bit hashes instead of author names to save memory (collision probability is negligible)
        # ------------------------------------------------------------
        print("[1/3] First scan: counting reviews per user/hotel (this takes a long time)")
        author_hashes = array("q")
        hotel_cnt = Counter()
        total = placeholder = 0
        for rec in iter_records(args.src):
            total += 1
            if not is_real_author(rec.get("author")):      # anonymous authors are excluded from the counts
                placeholder += 1
                continue
            author_hashes.append(hash(rec.get("author")))
            hotel_cnt[rec.get("hotel_url", "")] += 1
            if total % 5_000_000 == 0:
                print(f"   ...{total:,} lines")
        n_bad_json = iter_records.bad

        ah = np.frombuffer(author_hashes, dtype=np.int64)
        uniq, cnts = np.unique(ah, return_counts=True)
        keep_author_hash = set(uniq[cnts >= args.min_user].tolist())
        keep_hotel = {h for h, c in hotel_cnt.items() if c >= args.min_hotel}
        report.append({"stage": "0a_raw (full original)", "rows": total,
                       "users": len(uniq) + (1 if placeholder else 0), "hotels": len(hotel_cnt)})
        report.append({"stage": "0b_remove reviews by anonymous authors ('/undefined', 'null', etc.)", "rows": total - placeholder,
                       "users": len(uniq), "hotels": len(hotel_cnt),
                       "density_%": round(100 * (total - placeholder) / (len(uniq) * len(hotel_cnt)), 8),
                       "avg_reviews_per_user": round((total - placeholder) / len(uniq), 2)})
        print(f"   original {total:,} lines ({n_bad_json} JSON errors), {placeholder:,} lines by anonymous authors excluded, "
              f"{len(uniq):,} users, {len(hotel_cnt):,} hotels")
        print(f"   first-pass candidates: {len(keep_author_hash):,} users, {len(keep_hotel):,} hotels")
        del ah, uniq, cnts, author_hashes

        # ------------------------------------------------------------
        # Second scan: save only candidate rows to the interim file (text included)
        # ------------------------------------------------------------
        print("[2/3] Second scan: extracting candidate rows")
        cols = ["row_id", "hotel_url", "author", "date", "rating", *ASPECT_COLS, "title", "text"]
        kept = 0
        with open(interim, "w", newline="", encoding="utf-8") as fo:
            w = csv.writer(fo)
            w.writerow(cols)
            for i, rec in enumerate(iter_records(args.src)):
                if not is_real_author(rec.get("author")):
                    continue
                if hash(rec.get("author")) not in keep_author_hash:
                    continue
                if rec.get("hotel_url", "") not in keep_hotel:
                    continue
                pd_ = rec.get("property_dict") or {}
                w.writerow([i, rec.get("hotel_url"), rec.get("author"), rec.get("date"),
                            rec.get("rating"), *[pd_.get(k) for k in ASPECTS],
                            rec.get("title", ""), rec.get("text", "")])
                kept += 1
        print(f"   {kept:,} candidate rows saved -> {interim}")
        json.dump(report, open(raw_stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ------------------------------------------------------------
    # 3) In-memory processing (load only the key columns first and filter)
    # ------------------------------------------------------------
    print("[3/3] Cleaning / duplicate removal / iterative filtering")
    key_cols = ["row_id", "hotel_url", "author", "date", "rating", *ASPECT_COLS]
    df = pd.read_csv(interim, usecols=key_cols)
    report.append(stats(df, "1_first-pass candidates (pre-filter on minimum reviews per user/hotel)"))

    # format validation: rating 1~5, date conversion, missing author/hotel
    n0 = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df.rating.between(1, 5) & df.date.notna()
            & df.author.notna() & df.hotel_url.notna()]
    for c in ASPECT_COLS:                      # sub-ratings outside 1~5 are set to missing
        df.loc[~df[c].between(1, 5), c] = np.nan
    print(f"   format errors removed: {n0 - len(df):,} rows")
    report.append(stats(df, "2_format validation (rating 1~5, date conversion)"))

    # D. for duplicate (user, hotel) pairs keep only the most recent review
    n0 = len(df)
    df = (df.sort_values(["date", "row_id"])
            .drop_duplicates(["author", "hotel_url"], keep="last"))
    print(f"   duplicates removed (latest kept): {n0 - len(df):,} rows")
    report.append(stats(df, "3_duplicate removal (latest 1 per user-hotel)"))

    # B. minimum reviews per user/hotel: repeat until nothing more is removed
    df, it = kcore(df, args.min_user, args.min_hotel)
    print(f"   iterative filtering, {it} rounds: users >= {args.min_user}, hotels >= {args.min_hotel} reviews -> {len(df):,} rows")
    report.append(stats(df, f"4_iterative filtering (users>={args.min_user}, hotels>={args.min_hotel})"))

    # if the data is too large, sample users at random and filter iteratively again (proposal: sampling is used)
    if args.max_users and df.author.nunique() > args.max_users:
        rng = np.random.default_rng(args.seed)
        picked = set(rng.choice(df.author.unique(), size=args.max_users, replace=False))
        df = df[df.author.isin(picked)]
        df, it = kcore(df, args.min_user, args.min_hotel_after_sampling)
        print(f"   after sampling {args.max_users:,} users, refiltering ({it} rounds, users>={args.min_user}, "
              f"hotels>={args.min_hotel_after_sampling}) -> {len(df):,} rows")
        report.append(stats(df, f"4b_user sampling (max {args.max_users:,} users, seed={args.seed}) + "
                                f"refiltering (hotels>={args.min_hotel_after_sampling})"))

    # ID assignment (assigned once on the union so both files use the same IDs)
    df["user_id"] = pd.factorize(df.author)[0]
    df["hotel_id"] = pd.factorize(df.hotel_url)[0]

    # C. split by whether all 6 sub-ratings are present
    df["has_all_aspects"] = df[ASPECT_COLS].notna().all(axis=1)
    full_ids = set(df.loc[df.has_all_aspects, "row_id"])
    part_ids = set(df.loc[~df.has_all_aspects, "row_id"])
    report.append(stats(df[df.has_all_aspects], "5a_hotelrec_train (all 6 aspects present)"))
    report.append(stats(df[~df.has_all_aspects], "5b_hotelrec_partial (some/no aspects)"))

    # save the mapping tables
    hm = df.drop_duplicates("hotel_id")[["hotel_id", "hotel_url"]].copy()
    ex = hm.hotel_url.str.extract(URL_RE)
    hm["tripadvisor_id"] = ex["d"]
    hm["hotel_name"] = ex["name"].str.replace("_", " ")
    hm["location"] = ex["loc"].str.replace("_", " ")
    hm.sort_values("hotel_id").to_csv(out / "hotel_map.csv", index=False, encoding="utf-8-sig")
    (df.drop_duplicates("user_id")[["user_id", "author"]].sort_values("user_id")
       .to_csv(out / "user_map.csv", index=False, encoding="utf-8-sig"))

    free_gb = shutil.disk_usage(out).free / 1e9
    print(f"   {len(df):,} rows to save, free disk space {free_gb:.1f}GB")
    if free_gb < 3:
        print("   [Warning] Less than 3GB of free space. Saving may be interrupted.")

    # final save: reread the interim file in chunks and attach title/text to the selected rows
    meta = df.set_index("row_id")[["user_id", "hotel_id", "date"]]
    f_train = open(out / "hotelrec_train.csv", "w", newline="", encoding="utf-8-sig")
    f_part = open(out / "hotelrec_partial.csv", "w", newline="", encoding="utf-8-sig")
    out_cols = ["user_id", "hotel_id", "rating", "date", *ASPECT_COLS, "title", "text"]
    for fh in (f_train, f_part):                 # the header is written once at the start
        pd.DataFrame(columns=out_cols).to_csv(fh, index=False)
    for ch in pd.read_csv(interim, chunksize=200_000):
        ch = ch[ch.row_id.isin(meta.index)].copy()
        if ch.empty:
            continue
        ch["user_id"] = ch.row_id.map(meta.user_id)
        ch["hotel_id"] = ch.row_id.map(meta.hotel_id)
        ch["date"] = ch.row_id.map(meta.date).dt.strftime("%Y-%m-%d")
        ch["title"] = ch["title"].fillna("")
        ch["text"] = ch["text"].fillna("")
        for c in ASPECT_COLS:                    # as in the cleaning step, sub-ratings outside 1~5 are set to missing
            ch.loc[~ch[c].between(1, 5), c] = np.nan
        t = ch[ch.row_id.isin(full_ids)][out_cols]
        p = ch[ch.row_id.isin(part_ids)][out_cols]
        t.to_csv(f_train, header=False, index=False)
        p.to_csv(f_part, header=False, index=False)
    f_train.close()
    f_part.close()

    rep = pd.DataFrame(report)
    rep.to_csv(out / "reports" / "hotelrec_report.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n==== Before/after preprocessing ====")
    print(rep.to_string(index=False))
    print(f"\nSaved: {out}")
    print("interim_candidates.csv is an intermediate file and can be deleted after checking.")


if __name__ == "__main__":
    main()
