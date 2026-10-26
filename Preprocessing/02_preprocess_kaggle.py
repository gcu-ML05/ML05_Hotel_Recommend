"""
Kaggle "515K Hotel Reviews Data in Europe" (Booking.com, 테스트용) 전처리

입력 : Hotel_Reviews.csv
출력 : data/processing/kaggle/
    kaggle_test.csv                 정제된 리뷰 (테스트용)
    hotel_meta_kaggle.csv           호텔 정보 (hotel_id, 이름, 주소, 도시, 좌표, 평균 점수)
    reports/kaggle_report.csv       전처리 전후 비교
    reports/kaggle_traveler_type_counts.csv   여행자 유형 분포 (페르소나 검증용)

팀 결정 사항
    E. Reviewer_Score(10점 척도)를 2로 나눠 5점 척도 rating으로 변환 (원본 점수도 함께 보관)

특징
    - 사용자 ID가 없음 -> 협업 필터링 테스트는 불가, 콘텐츠 기반/여행자 유형 검증용
    - Tags에서 여행 목적, 여행자 유형, 숙박일수, 객실 유형, 모바일 여부를 추출

사용법
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
PLACEHOLDERS = {"No Negative", "No Positive"}   # 리뷰 내용이 없다는 뜻의 자리표시 문구


def city_from_address(addr):
    parts = str(addr).split()
    return COUNTRY_TO_CITY.get(parts[-1]) or COUNTRY_TO_CITY.get(" ".join(parts[-2:]))


def parse_tags(tag_str):
    """Tags 문자열 -> (여행 목적, 여행자 유형, 숙박일수, 객실 유형, 모바일, 반려동물)"""
    trip = traveler = room = None
    nights = np.nan
    mobile = pet = False
    try:
        tags = [t.strip() for t in ast.literal_eval(tag_str)]
    except (ValueError, SyntaxError):
        return trip, traveler, nights, room, mobile, pet, True   # 마지막 값: 파싱 실패 표시
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
            rooms.append(t)               # 나머지는 객실 유형으로 간주
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
    print(f"원본: {df.shape[0]:,}행 x {df.shape[1]}열")
    report.append(summary(df, "0_raw(원본)", {
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_lat_lng": int(df.lat.isna().sum()),
        "positive_placeholder": int((df.Positive_Review.str.strip() == "No Positive").sum()),
        "negative_placeholder": int((df.Negative_Review.str.strip() == "No Negative").sum()),
    }))

    # 1) 완전 중복 행 제거
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"완전 중복 제거: {n0 - len(df):,}행")
    report.append(summary(df, "1_중복제거(완전 중복 행)"))

    # 2) 공백 정리, 날짜/호텔/도시
    df["Reviewer_Nationality"] = df["Reviewer_Nationality"].astype(str).str.strip()
    df["review_date"] = pd.to_datetime(df["Review_Date"], format="%m/%d/%Y", errors="coerce")
    n_bad_date = int(df.review_date.isna().sum())
    df["city"] = df["Hotel_Address"].apply(city_from_address)
    df["hotel_id"] = df.groupby(["Hotel_Name", "Hotel_Address"], sort=False).ngroup()

    # 3) 평점: 10점 -> 5점 (팀 결정 E)
    df["reviewer_score_10"] = df["Reviewer_Score"]
    df["rating"] = (df["Reviewer_Score"] / 2).round(2)

    # 4) 리뷰 텍스트: 자리표시 문구 제거 후 긍정+부정 결합
    df["positive_review"] = df["Positive_Review"].apply(clean_review)
    df["negative_review"] = df["Negative_Review"].apply(clean_review)
    df["review_text"] = (df.positive_review + " " + df.negative_review).str.strip()

    # 5) Tags 파싱
    parsed = df["Tags"].apply(parse_tags)
    tag_df = pd.DataFrame(parsed.tolist(),
                          columns=["trip_type", "traveler_type", "nights", "room_type",
                                   "is_mobile", "with_pet", "tag_parse_failed"])
    df = pd.concat([df.reset_index(drop=True), tag_df], axis=1)
    n_parse_fail = int(df.tag_parse_failed.sum())

    # 6) 검증: 형식 오류 행 제거
    n0 = len(df)
    df = df[df.review_date.notna() & df.rating.between(1, 5) & df.city.notna()]
    print(f"형식 오류 제거(날짜/평점/도시): {n0 - len(df):,}행 "
          f"(날짜 변환 실패 {n_bad_date}, Tags 파싱 실패 {n_parse_fail})")

    final_cols = ["hotel_id", "Hotel_Name", "city", "review_date", "reviewer_score_10", "rating",
                  "Reviewer_Nationality", "trip_type", "traveler_type", "nights", "room_type",
                  "is_mobile", "with_pet", "positive_review", "negative_review", "review_text"]
    res = df[final_cols].rename(columns={"Hotel_Name": "hotel_name",
                                         "Reviewer_Nationality": "reviewer_nationality"})
    res.insert(0, "review_id", range(len(res)))
    res["review_date"] = res.review_date.dt.strftime("%Y-%m-%d")

    report.append(summary(res, "2_최종(전처리 후)", {
        "missing_trip_type": int(res.trip_type.isna().sum()),
        "missing_traveler_type": int(res.traveler_type.isna().sum()),
        "missing_room_type": int(res.room_type.isna().sum()),
        "missing_nights": int(res.nights.isna().sum()),
        "empty_review_text": int((res.review_text == "").sum()),
        "positive_empty": int((res.positive_review == "").sum()),
        "negative_empty": int((res.negative_review == "").sum()),
        "tag_parse_failed": n_parse_fail,
    }))

    # 호텔 메타 / 저장
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
    tcount = res.traveler_type.fillna("(없음)").value_counts().rename_axis("traveler_type").reset_index(name="count")
    tcount.to_csv(out / "reports" / "kaggle_traveler_type_counts.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n==== 전처리 전후 비교 ====")
    print(rep.T.to_string(header=False))
    print("\n여행자 유형 분포")
    print(tcount.to_string(index=False))
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()