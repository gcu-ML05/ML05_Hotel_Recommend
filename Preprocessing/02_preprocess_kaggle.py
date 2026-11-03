"""
Preprocessing of Kaggle "515K Hotel Reviews Data in Europe" (Booking.com, for testing)

Input : Hotel_Reviews.csv
Output : data/processing/kaggle/
    kaggle_test.csv                 cleaned reviews (for testing)
    hotel_meta_kaggle.csv           hotel info (hotel_id, name, address, city, coordinates, average score)
    reports/kaggle_report.csv       before/after preprocessing comparison
    reports/kaggle_traveler_type_counts.csv   traveler type distribution (for persona validation)

Team decisions
    E. Divide Reviewer_Score (10-point scale) by 2 to get a 5-point rating (the original score is kept too)

Notes
    - No user IDs -> collaborative filtering cannot be tested; used for content-based / traveler type validation
    - Trip purpose, traveler type, number of nights, room type and mobile flag are extracted from Tags

Usage
    python 02_preprocess_kaggle.py --src ../data/raw/Hotel_Reviews.csv
"""
import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRY_TO_CITY = {"France": "Paris", "United Kingdom": "London", "Spain": "Barcelona",
                   "Italy": "Milan", "Austria": "Vienna", "Netherlands": "Amsterdam"}

TRIP_TYPES = {"Leisure trip", "Business trip"}
TRAVELER_TYPES = {"Couple", "Solo traveler", "Family with young children",
                  "Family with older children", "Group", "Travelers with friends"}
MOBILE_TAG = "Submitted from a mobile device"
PET_TAG = "With a pet"
NIGHTS_RE = re.compile(r"^Stayed (\d+) nights?$")
PLACEHOLDERS = {"No Negative", "No Positive"}   # placeholder phrases meaning the review has no content


def city_from_address(addr):
    parts = str(addr).split()
    return COUNTRY_TO_CITY.get(parts[-1]) or COUNTRY_TO_CITY.get(" ".join(parts[-2:]))


def parse_tags(tag_str):
    """Tags string -> (trip purpose, traveler type, nights, room type, mobile, pet)"""
    trip = traveler = room = None
    nights = np.nan
    mobile = pet = False
    try:
        tags = [t.strip() for t in ast.literal_eval(tag_str)]
    except (ValueError, SyntaxError):
        return trip, traveler, nights, room, mobile, pet, True   # last value: parse failure flag
    rooms = []
    for t in tags:
        m = NIGHTS_RE.match(t)
        if t in TRIP_TYPES:
            trip = t
        elif t in TRAVELER_TYPES:
            traveler = t
        elif m:
            nights = int(m.group(1))
        elif t == MOBILE_TAG:
            mobile = True
        elif t == PET_TAG:
            pet = True
        else:
            rooms.append(t)               # everything else is treated as the room type
    room = " | ".join(rooms) if rooms else None
    return trip, traveler, nights, room, mobile, pet, False


def clean_review(s):
    s = "" if pd.isna(s) else str(s).strip()
    return "" if s in PLACEHOLDERS else s


def summary(df, stage, extra=None):
    row = {"stage": stage, "rows": len(df),
           "hotels": df[["Hotel_Name", "Hotel_Address"]].drop_duplicates().shape[0]
                     if "Hotel_Name" in df else df["hotel_id"].nunique()}
    row.update(extra or {})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/raw/Hotel_Reviews.csv")
    ap.add_argument("--out-dir", default="../data/processing/kaggle")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    report = []

    df = pd.read_csv(args.src)
    print(f"Original: {df.shape[0]:,} rows x {df.shape[1]} columns")
    report.append(summary(df, "0_raw (original)", {
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_lat_lng": int(df.lat.isna().sum()),
        "positive_placeholder": int((df.Positive_Review.str.strip() == "No Positive").sum()),
        "negative_placeholder": int((df.Negative_Review.str.strip() == "No Negative").sum()),
    }))

    # 1) remove exact duplicate rows
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Exact duplicates removed: {n0 - len(df):,} rows")
    report.append(summary(df, "1_duplicate removal (exact duplicate rows)"))

    # 2) whitespace cleanup, date/hotel/city
    df["Reviewer_Nationality"] = df["Reviewer_Nationality"].astype(str).str.strip()
    df["review_date"] = pd.to_datetime(df["Review_Date"], format="%m/%d/%Y", errors="coerce")
    n_bad_date = int(df.review_date.isna().sum())
    df["city"] = df["Hotel_Address"].apply(city_from_address)
    df["hotel_id"] = df.groupby(["Hotel_Name", "Hotel_Address"], sort=False).ngroup()

    # 3) rating: 10-point -> 5-point (team decision E)
    df["reviewer_score_10"] = df["Reviewer_Score"]
    df["rating"] = (df["Reviewer_Score"] / 2).round(2)

    # 4) review text: remove placeholder phrases, then join positive + negative
    df["positive_review"] = df["Positive_Review"].apply(clean_review)
    df["negative_review"] = df["Negative_Review"].apply(clean_review)
    df["review_text"] = (df.positive_review + " " + df.negative_review).str.strip()

    # 5) parse Tags
    parsed = df["Tags"].apply(parse_tags)
    tag_df = pd.DataFrame(parsed.tolist(),
                          columns=["trip_type", "traveler_type", "nights", "room_type",
                                   "is_mobile", "with_pet", "tag_parse_failed"])
    df = pd.concat([df.reset_index(drop=True), tag_df], axis=1)
    n_parse_fail = int(df.tag_parse_failed.sum())

    # 6) validation: remove rows with format errors
    n0 = len(df)
    df = df[df.review_date.notna() & df.rating.between(1, 5) & df.city.notna()]
    print(f"Format errors removed (date/rating/city): {n0 - len(df):,} rows "
          f"(date conversion failed {n_bad_date}, Tags parse failed {n_parse_fail})")

    final_cols = ["hotel_id", "Hotel_Name", "city", "review_date", "reviewer_score_10", "rating",
                  "Reviewer_Nationality", "trip_type", "traveler_type", "nights", "room_type",
                  "is_mobile", "with_pet", "positive_review", "negative_review", "review_text"]
    res = df[final_cols].rename(columns={"Hotel_Name": "hotel_name",
                                         "Reviewer_Nationality": "reviewer_nationality"})
    res.insert(0, "review_id", range(len(res)))
    res["review_date"] = res.review_date.dt.strftime("%Y-%m-%d")

    report.append(summary(res, "2_final (after preprocessing)", {
        "missing_trip_type": int(res.trip_type.isna().sum()),
        "missing_traveler_type": int(res.traveler_type.isna().sum()),
        "missing_room_type": int(res.room_type.isna().sum()),
        "missing_nights": int(res.nights.isna().sum()),
        "empty_review_text": int((res.review_text == "").sum()),
        "positive_empty": int((res.positive_review == "").sum()),
        "negative_empty": int((res.negative_review == "").sum()),
        "tag_parse_failed": n_parse_fail,
    }))

    # hotel meta / save
    meta = (df.groupby("hotel_id")
              .agg(hotel_name=("Hotel_Name", "first"), address=("Hotel_Address", "first"),
                   city=("city", "first"), lat=("lat", "first"), lng=("lng", "first"),
                   average_score=("Average_Score", "first"),
                   total_number_of_reviews=("Total_Number_of_Reviews", "first"),
                   n_reviews_in_data=("hotel_id", "size")).reset_index())
    meta.to_csv(out / "hotel_meta_kaggle.csv", index=False, encoding="utf-8-sig")
    res.to_csv(out / "kaggle_test.csv", index=False, encoding="utf-8-sig")

    rep = pd.DataFrame(report)
    rep.to_csv(out / "reports" / "kaggle_report.csv", index=False, encoding="utf-8-sig")
    tcount = res.traveler_type.fillna("(none)").value_counts().rename_axis("traveler_type").reset_index(name="count")
    tcount.to_csv(out / "reports" / "kaggle_traveler_type_counts.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n==== Before/after preprocessing ====")
    print(rep.T.to_string(header=False))
    print("\nTraveler type distribution")
    print(tcount.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
