"""
HotelRec (학습용) 전처리

입력 : HotelRec.txt  (한 줄에 JSON 하나인 JSON Lines, 약 5천만 리뷰)
출력 : data/processing/hotelrec/
    hotelrec_train.csv      6개 세부 평점이 모두 있는 리뷰 (모델 학습용)
    hotelrec_partial.csv    세부 평점이 일부/전부 없는 리뷰 (제거 대신 분리 저장)
    hotel_map.csv           hotel_id <-> hotel_url / 호텔명 / 지역
    user_map.csv            user_id <-> author
    reports/hotelrec_report.csv   단계별 전후 비교 (행/사용자/호텔 수, 결측, 희소도)

팀 결정 사항
    A. 지역 제한 없음 (전 세계)
    B. 사용자 MIN_USER 건 이상, 호텔 MIN_HOTEL 건 이상 (반복 필터링, k-core)
    C. 6개 세부 평점이 모두 있는 리뷰만 hotelrec_train.csv, 나머지는 hotelrec_partial.csv로 분리 저장
    D. 같은 (사용자, 호텔) 중복은 가장 최근 리뷰만 유지
    (E는 Kaggle 전처리에서 적용)

처리 순서 (메모리를 아끼려고 원본을 한 줄씩 읽음)
    1) 1차 스캔 : 원본 전체를 읽어 사용자별/호텔별 리뷰 수 집계
    2) 2차 스캔 : 후보(사용자 MIN_USER 건 이상, 호텔 MIN_HOTEL 건 이상) 행만 interim 파일로 저장
    3) 메모리 처리 : 형식 검증 -> 중복 제거(최신) -> 반복 필터링 -> ID 부여 -> 분리 저장

사용법
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

# ---------------- 기본 설정 (옵션으로 변경 가능) ----------------
MIN_USER = 5      # 사용자 최소 리뷰 수
MIN_HOTEL = 20    # 호텔 최소 리뷰 수
# ---------------------------------------------------------------

ASPECTS = {  # property_dict 키 -> 컬럼명 (제안서의 6개 aspect)
    "service": "service", "cleanliness": "cleanliness", "value": "value",
    "rooms": "rooms", "location": "location", "sleep quality": "sleep_quality",
}
ASPECT_COLS = list(ASPECTS.values())
# 실제 사용자가 아닌 익명/자리표시자 작성자 (하나의 사용자로 묶이면 학습이 왜곡되므로 제거)
PLACEHOLDER_AUTHORS = {"/undefined", "undefined", "null", "none", "nan", "",
                       "a tripadvisor member", "lonely planet traveler"}
URL_RE = re.compile(r"-d(?P<d>\d+)-Reviews-(?P<name>.+)-(?P<loc>[^-]+)\.html$")


def iter_records(path):
    """JSON Lines를 한 줄씩 읽어 dict로 반환 (깨진 줄은 건너뜀)"""
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
    """단계별 요약 통계 한 줄"""
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
    """사용자/호텔 최소 리뷰 수를 만족할 때까지 반복해서 걸러냄"""
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
                    help="반복 필터링 후 무작위로 남길 최대 사용자 수 (0이면 샘플링 안 함)")
    ap.add_argument("--min-hotel-after-sampling", type=int, default=10,
                    help="샘플링 후 재필터링에 쓰는 호텔 최소 리뷰 수 (사용자를 줄이면 호텔당 리뷰도 줄어 기준을 완화)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="1~2차 스캔을 건너뛰고 기존 interim 파일과 reports/raw_stats.json을 사용")
    ap.add_argument("--interim", default=None, help="interim 파일 경로 (기본: out-dir/interim_candidates.csv)")
    # raw_stats.json이 없을 때(이전 버전으로 1~2차 스캔을 이미 돌린 경우) 로그에 찍힌 원본 통계를 직접 입력
    ap.add_argument("--raw-total", type=int, help="원본 전체 줄 수 (예: 50264531)")
    ap.add_argument("--raw-placeholder", type=int, help="제외된 익명 작성자 줄 수 (예: 1216923)")
    ap.add_argument("--raw-users", type=int, help="익명 제외 후 사용자 수 (예: 21891400)")
    ap.add_argument("--raw-hotels", type=int, help="호텔 수 (예: 365055)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    interim = Path(args.interim) if args.interim else out / "interim_candidates.csv"
    raw_stats_path = out / "reports" / "raw_stats.json"
    report = []
    print(f"저장 위치: {out.resolve()}  (디스크 여유 공간 {shutil.disk_usage(out).free / 1e9:.1f}GB)")

    if args.resume:
        print("[1~2/3] --resume: 스캔을 건너뛰고 기존 interim 파일을 사용합니다")
        if raw_stats_path.exists():
            report = json.load(open(raw_stats_path, encoding="utf-8"))
        elif None not in (args.raw_total, args.raw_placeholder, args.raw_users, args.raw_hotels):
            tot, ph, nu, nh = args.raw_total, args.raw_placeholder, args.raw_users, args.raw_hotels
            report = [
                {"stage": "0a_raw(원본 전체)", "rows": tot,
                 "users": nu + (1 if ph else 0), "hotels": nh},
                {"stage": "0b_익명 작성자('/undefined', 'null' 등) 리뷰 제거", "rows": tot - ph,
                 "users": nu, "hotels": nh,
                 "density_%": round(100 * (tot - ph) / (nu * nh), 8),
                 "avg_reviews_per_user": round((tot - ph) / nu, 2)},
            ]
        else:
            raise SystemExit("raw_stats.json이 없습니다. --raw-total, --raw-placeholder, "
                             "--raw-users, --raw-hotels 값을 함께 입력하세요 (1차 스캔 로그에 출력된 숫자).")
        if not interim.exists():
            raise SystemExit(f"interim 파일을 찾을 수 없습니다: {interim}  (--interim 으로 경로를 지정하세요)")
    else:
        # ------------------------------------------------------------
        # 1차 스캔: 사용자/호텔별 리뷰 수 집계
        #   작성자 이름 대신 64비트 해시를 저장해 메모리를 아낌 (충돌 확률은 무시 가능)
        # ------------------------------------------------------------
        print("[1/3] 1차 스캔: 사용자/호텔별 리뷰 수 집계 (시간이 오래 걸립니다)")
        author_hashes = array("q")
        hotel_cnt = Counter()
        total = placeholder = 0
        for rec in iter_records(args.src):
            total += 1
            if not is_real_author(rec.get("author")):      # 익명 작성자는 집계에서 제외
                placeholder += 1
                continue
            author_hashes.append(hash(rec.get("author")))
            hotel_cnt[rec.get("hotel_url", "")] += 1
            if total % 5_000_000 == 0:
                print(f"   ...{total:,}줄")
        n_bad_json = iter_records.bad

        ah = np.frombuffer(author_hashes, dtype=np.int64)
        uniq, cnts = np.unique(ah, return_counts=True)
        keep_author_hash = set(uniq[cnts >= args.min_user].tolist())
        keep_hotel = {h for h, c in hotel_cnt.items() if c >= args.min_hotel}
        report.append({"stage": "0a_raw(원본 전체)", "rows": total,
                       "users": len(uniq) + (1 if placeholder else 0), "hotels": len(hotel_cnt)})
        report.append({"stage": "0b_익명 작성자('/undefined', 'null' 등) 리뷰 제거", "rows": total - placeholder,
                       "users": len(uniq), "hotels": len(hotel_cnt),
                       "density_%": round(100 * (total - placeholder) / (len(uniq) * len(hotel_cnt)), 8),
                       "avg_reviews_per_user": round((total - placeholder) / len(uniq), 2)})
        print(f"   원본 {total:,}줄 (JSON 오류 {n_bad_json}줄), 익명 작성자 {placeholder:,}줄 제외, "
              f"사용자 {len(uniq):,}명, 호텔 {len(hotel_cnt):,}개")
        print(f"   1차 후보: 사용자 {len(keep_author_hash):,}명, 호텔 {len(keep_hotel):,}개")
        del ah, uniq, cnts, author_hashes

        # ------------------------------------------------------------
        # 2차 스캔: 후보 행만 interim 파일로 저장 (텍스트 포함)
        # ------------------------------------------------------------
        print("[2/3] 2차 스캔: 후보 행 추출")
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
        print(f"   후보 {kept:,}행 저장 -> {interim}")
        json.dump(report, open(raw_stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ------------------------------------------------------------
    # 3) 메모리 처리 (키 컬럼만 먼저 로드해서 필터링)
    # ------------------------------------------------------------
    print("[3/3] 정제 / 중복 제거 / 반복 필터링")
    key_cols = ["row_id", "hotel_url", "author", "date", "rating", *ASPECT_COLS]
    df = pd.read_csv(interim, usecols=key_cols)
    report.append(stats(df, "1_1차후보(사용자/호텔 최소 리뷰 수 사전 필터)"))

    # 형식 검증: 평점 1~5, 날짜 변환, 작성자/호텔 결측
    n0 = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df.rating.between(1, 5) & df.date.notna()
            & df.author.notna() & df.hotel_url.notna()]
    for c in ASPECT_COLS:                      # 세부 평점이 1~5 밖이면 결측 처리
        df.loc[~df[c].between(1, 5), c] = np.nan
    print(f"   형식 오류 제거: {n0 - len(df):,}행")
    report.append(stats(df, "2_형식검증(평점 1~5, 날짜 변환)"))

    # D. 같은 (사용자, 호텔) 중복은 가장 최근 리뷰만 유지
    n0 = len(df)
    df = (df.sort_values(["date", "row_id"])
            .drop_duplicates(["author", "hotel_url"], keep="last"))
    print(f"   중복 제거(최신만 유지): {n0 - len(df):,}행")
    report.append(stats(df, "3_중복제거(사용자-호텔 최신 1건)"))

    # B. 사용자/호텔 최소 리뷰 수: 더 이상 줄지 않을 때까지 반복
    df, it = kcore(df, args.min_user, args.min_hotel)
    print(f"   반복 필터링 {it}회: 사용자 {args.min_user}건 이상, 호텔 {args.min_hotel}건 이상 -> {len(df):,}행")
    report.append(stats(df, f"4_반복필터링(사용자>={args.min_user}, 호텔>={args.min_hotel})"))

    # 데이터가 너무 크면 사용자를 무작위로 샘플링한 뒤 다시 반복 필터링 (제안서: 샘플링 사용)
    if args.max_users and df.author.nunique() > args.max_users:
        rng = np.random.default_rng(args.seed)
        picked = set(rng.choice(df.author.unique(), size=args.max_users, replace=False))
        df = df[df.author.isin(picked)]
        df, it = kcore(df, args.min_user, args.min_hotel_after_sampling)
        print(f"   사용자 {args.max_users:,}명 샘플링 후 재필터링({it}회, 사용자>={args.min_user}, "
              f"호텔>={args.min_hotel_after_sampling}) -> {len(df):,}행")
        report.append(stats(df, f"4b_사용자 샘플링(최대 {args.max_users:,}명, seed={args.seed}) + "
                                f"재필터링(호텔>={args.min_hotel_after_sampling})"))

    # ID 부여 (union 기준으로 한 번만 부여해서 두 파일에서 같은 ID를 쓰도록 함)
    df["user_id"] = pd.factorize(df.author)[0]
    df["hotel_id"] = pd.factorize(df.hotel_url)[0]

    # C. 6개 세부 평점이 모두 있는지로 분리
    df["has_all_aspects"] = df[ASPECT_COLS].notna().all(axis=1)
    full_ids = set(df.loc[df.has_all_aspects, "row_id"])
    part_ids = set(df.loc[~df.has_all_aspects, "row_id"])
    report.append(stats(df[df.has_all_aspects], "5a_hotelrec_train(6개 aspect 모두 있음)"))
    report.append(stats(df[~df.has_all_aspects], "5b_hotelrec_partial(aspect 일부/없음)"))

    # 매핑표 저장
    hm = df.drop_duplicates("hotel_id")[["hotel_id", "hotel_url"]].copy()
    ex = hm.hotel_url.str.extract(URL_RE)
    hm["tripadvisor_id"] = ex["d"]
    hm["hotel_name"] = ex["name"].str.replace("_", " ")
    hm["location"] = ex["loc"].str.replace("_", " ")
    hm.sort_values("hotel_id").to_csv(out / "hotel_map.csv", index=False, encoding="utf-8-sig")
    (df.drop_duplicates("user_id")[["user_id", "author"]].sort_values("user_id")
       .to_csv(out / "user_map.csv", index=False, encoding="utf-8-sig"))

    free_gb = shutil.disk_usage(out).free / 1e9
    print(f"   최종 저장 대상 {len(df):,}행, 디스크 여유 공간 {free_gb:.1f}GB")
    if free_gb < 3:
        print("   [경고] 여유 공간이 3GB 미만입니다. 저장 중 중단될 수 있습니다.")

    # 최종 저장: interim을 청크로 다시 읽어 선택된 행에 title/text 붙여서 저장
    meta = df.set_index("row_id")[["user_id", "hotel_id", "date"]]
    f_train = open(out / "hotelrec_train.csv", "w", newline="", encoding="utf-8-sig")
    f_part = open(out / "hotelrec_partial.csv", "w", newline="", encoding="utf-8-sig")
    out_cols = ["user_id", "hotel_id", "rating", "date", *ASPECT_COLS, "title", "text"]
    for fh in (f_train, f_part):                 # 헤더는 처음에 한 번만 기록
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
        for c in ASPECT_COLS:                    # 정제 단계와 동일하게 범위(1~5) 밖 세부 평점은 결측 처리
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
    print("\n==== 전처리 전후 비교 ====")
    print(rep.to_string(index=False))
    print(f"\n저장 완료: {out}")
    print("interim_candidates.csv 는 중간 파일이라 확인 후 삭제해도 됩니다.")


if __name__ == "__main__":
    main()