"""
전처리 결과 검증 (HotelRec 학습용 + Kaggle 테스트용)

전처리 스크립트가 의도대로 동작했는지 자동으로 확인하고 PASS/FAIL 목록과 요약 통계를 출력합니다.
결과는 각 reports 폴더의 validation_report.csv 로도 저장됩니다.

사용법
    python 03_validate_preprocessing.py --hotelrec-dir D:/hotelrec --kaggle-dir ../data/processing/kaggle
    (HotelRec 전처리 때 --min-user/--min-hotel 을 바꿨다면 같은 값을 넘겨주세요.
     --max-users 샘플링을 썼다면 --min-hotel 은 샘플링 후 기준(기본 10)을 넣으세요.)
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
    print(f"\n=== HotelRec 검증: {d} ===")
    files = ["hotelrec_train.csv", "hotelrec_partial.csv", "user_map.csv", "hotel_map.csv"]
    missing = [f for f in files if not (d / f).exists()]
    check("필수 파일 존재", not missing, f"없는 파일: {missing}" if missing else "")
    if missing:
        return

    cols = ["user_id", "hotel_id", "rating", "date", *ASPECTS, "title", "text"]
    t = pd.read_csv(d / "hotelrec_train.csv")
    p = pd.read_csv(d / "hotelrec_partial.csv")
    um = pd.read_csv(d / "user_map.csv")
    hm = pd.read_csv(d / "hotel_map.csv")
    check("컬럼 구성(train/partial)", list(t.columns) == cols and list(p.columns) == cols)

    a = pd.concat([t, p], ignore_index=True)
    print(f"  train {len(t):,}행 / partial {len(p):,}행 / 합계 {len(a):,}행")

    # C. 분리 기준
    check("train: 6개 세부 평점 결측 없음", t[ASPECTS].notna().all().all(),
          f"결측 {int(t[ASPECTS].isna().sum().sum())}개")
    check("partial: 모든 행에 세부 평점 결측 1개 이상", p[ASPECTS].isna().any(axis=1).all(),
          f"완전한 행 {int(p[ASPECTS].notna().all(axis=1).sum())}개")

    # D. 중복
    check("(사용자,호텔) 중복 없음", not a.duplicated(["user_id", "hotel_id"]).any(),
          f"중복 {int(a.duplicated(['user_id', 'hotel_id']).sum())}건")

    # 값 범위 / 결측
    check("rating 1~5, 결측 없음", a.rating.between(1, 5).all(), f"결측 {int(a.rating.isna().sum())}")
    asp_ok = all(a[c].dropna().between(1, 5).all() for c in ASPECTS)
    check("세부 평점 값이 1~5 범위", asp_ok)
    dt = pd.to_datetime(a.date, errors="coerce")
    check("date 변환 가능", dt.notna().all(), f"범위 {dt.min().date()} ~ {dt.max().date()}")

    # B. 최소 리뷰 수
    uc, hc = a.user_id.value_counts(), a.hotel_id.value_counts()
    check(f"사용자 최소 리뷰 수 >= {min_user}", uc.min() >= min_user, f"최소 {uc.min()}건")
    check(f"호텔 최소 리뷰 수 >= {min_hotel}", hc.min() >= min_hotel, f"최소 {hc.min()}건")

    # ID 일관성
    check("user_id가 user_map에 모두 존재", set(a.user_id) <= set(um.user_id))
    check("hotel_id가 hotel_map에 모두 존재", set(a.hotel_id) <= set(hm.hotel_id))
    check("user_id/hotel_id 유일(map 파일)", um.user_id.is_unique and hm.hotel_id.is_unique)
    ph = um.author.fillna("").astype(str).str.strip().str.lower().isin(PLACEHOLDER_AUTHORS)
    check("익명 작성자('/undefined' 등) 없음", not ph.any(), f"{int(ph.sum())}명")

    # 텍스트
    et = int((a.text.fillna("").str.strip() == "").sum())
    print(f"  (참고) 본문 비어 있는 리뷰: {et:,}건 ({et / len(a):.2%})")

    # 모델링에 중요한 참고 지표: train 사용자당 리뷰 수 분포
    c = t.user_id.value_counts()
    print("\n  [참고] train(세부 평점 완전) 사용자당 리뷰 수 분포")
    for n in [1, 3, 5, 7, 10]:
        print(f"    {n}건 이상: {(c >= n).sum():>8,}명 ({(c >= n).mean():6.1%})")
    print(f"    평균 {c.mean():.2f}건, 중앙값 {c.median():.0f}건, 사용자 {len(c):,}명")
    sp = 1 - len(a) / (a.user_id.nunique() * a.hotel_id.nunique())
    print(f"  [참고] 전체(train+partial) 희소도 {sp:.4%}")


def validate_kaggle(d):
    d = Path(d)
    print(f"\n=== Kaggle 검증: {d} ===")
    f, mf = d / "kaggle_test.csv", d / "hotel_meta_kaggle.csv"
    missing = [x.name for x in (f, mf) if not x.exists()]
    check("필수 파일 존재", not missing, f"없는 파일: {missing}" if missing else "")
    if missing:
        return
    k = pd.read_csv(f, keep_default_na=False, na_values=[""])
    meta = pd.read_csv(mf)
    print(f"  {len(k):,}행, 호텔 {k.hotel_id.nunique():,}개")

    check("review_id 유일", k.review_id.is_unique)
    check("rating 1~5 범위, 결측 없음", k.rating.between(1, 5).all(), f"결측 {int(k.rating.isna().sum())}")
    check("rating = reviewer_score_10 / 2",
          np.allclose(k.rating, (k.reviewer_score_10 / 2).round(2), atol=0.011))
    check("review_date 변환 가능", pd.to_datetime(k.review_date, errors="coerce").notna().all(),
          f"{k.review_date.min()} ~ {k.review_date.max()}")
    # 열 전체가 자리표시 문구인 경우만 오류 (본문 중간에 우연히 같은 표현이 있는 것은 정상)
    check("자리표시 문구 없음(positive/negative 열이 'No Positive'/'No Negative'가 아님)",
          not (k.positive_review.eq("No Positive").any() or k.negative_review.eq("No Negative").any()))
    check("traveler_type 값이 허용 목록 안", k.traveler_type.dropna().isin(TRAVELER_TYPES).all())
    check("trip_type 값이 허용 목록 안", k.trip_type.dropna().isin(TRIP_TYPES).all(),
          f"결측 {int(k.trip_type.isna().sum()):,}건")
    check("모든 hotel_id가 hotel_meta에 존재", set(k.hotel_id) <= set(meta.hotel_id))
    # 짧은 리뷰("Just perfect" 등)는 서로 다른 사람이 같은 날 같은 점수로 쓸 수 있으므로 본문 40자 이상만 검사
    body = k.drop(columns="review_id")
    long_dup = int(body[body.review_text.fillna("").str.len() >= 40].duplicated().sum())
    check("중복 행 없음(본문 40자 이상)", long_dup == 0, f"중복 {long_dup}건")
    print(f"  (참고) 짧은 본문 포함 전체 중복 {int(body.duplicated().sum())}건")
    print(f"  (참고) 본문이 비어 있는 리뷰: {int(k.review_text.isna().sum()):,}건")
    print(f"  (참고) 여행자 유형 분포: {k.traveler_type.value_counts().to_dict()}")


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
    print(f"\n==== 검증 요약: {len(res)}개 중 PASS {len(res) - n_fail}개, FAIL {n_fail}개 ====")
    for d in (args.hotelrec_dir, args.kaggle_dir):
        rp = Path(d) / "reports"
        if rp.exists():
            res.to_csv(rp / "validation_report.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()