"""
6단계: HotelRec <-> Kaggle 호텔 매칭 평가 (신규 사용자 일반화)

Kaggle(Booking.com) 리뷰어는 HotelRec에 이력이 없는 신규 사용자이다. 두 데이터에 모두 있는 호텔에 대해
HotelRec으로 학습한 모델이 신규 사용자에게 내놓는 예측 점수가 Booking.com의 실제 점수와 얼마나 맞는지 본다.

(1) 호텔 매칭
    같은 도시 안에서 호텔명을 정규화(소문자, 기호 제거, 'hotel' 등 흔한 단어 제외)해 일치하면 매칭.
    일치하지 않으면 같은 도시 후보 중 문자열 유사도(difflib)가 MIN_RATIO 이상인 가장 비슷한 호텔과 매칭.
    HotelRec 호텔의 도시는 hotel_map.csv의 location(예: "London England")이 Kaggle 도시명으로 시작하는 것 중
    호텔이 가장 많은 location 하나로 정한다.

(2) 신규 사용자 예측 (사용자 이력이 없으므로 사용자 항은 0)
    GlobalMean          μ
    Baseline            μ + b_i
    MF                  μ + b_i                 (잠재 요인 p_u가 없으므로 호텔 편향까지만)
    Content(aspect)     w0 + Σ_k w_k · a_hk      (전역 aspect 가중치 x 호텔 aspect 평균)
    Hybrid              λ · MF + (1-λ) · Content (λ는 04단계에서 선택한 값)

(3) 평가
    리뷰 단위: Kaggle 리뷰 점수(rating = 10점/2)와의 RMSE, MAE, 스피어만 순위상관
    호텔 단위: 호텔별 Kaggle 평균 점수와의 피어슨/스피어만 상관 (리뷰어 개인차를 평균으로 줄인 비교)
    여행자 유형별 리뷰 단위 스피어만 상관

입력: data/pipeline/hotel_scores.npz (04단계 출력), hotel_aspects.npy, global_weights.npy
출력: Pipeline/outputs/06_*.csv, figures/06_*.png
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
    """반환: kaggle hotel_id, hotelrec hotel_id, 매칭 방식, 유사도"""
    rows = []
    for city, kc in km.groupby("city"):
        hc = hm[hm.location.str.match(rf"^{city}\b", case=False, na=False)]
        if hc.empty:
            continue
        # "London Ontario", "Paris Tennessee"처럼 이름만 같은 다른 도시를 빼기 위해 호텔이 가장 많은 location만 사용
        hc = hc[hc.location == hc.location.value_counts().index[0]]
        by_name = dict(zip(hc.key, hc.hotel_id))
        keys = list(by_name)
        for _, k in kc.iterrows():
            if k.key in by_name:
                rows.append((k.hotel_id, by_name[k.key], "exact", 1.0))
                continue
            # difflib.get_close_matches(문자열, 후보 목록, n=최대 개수, cutoff=최소 유사도): 비슷한 문자열 찾기
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

    # (1) 매칭
    m = match_hotels(hm, km)
    # HotelRec 호텔 하나에 Kaggle 호텔 여러 개가 붙으면 유사도가 가장 높은 것만 남기고,
    # 최고 유사도가 동점이면(예: "Hotel de Nell"과 "Nell Hotel Suites"가 모두 "nell") 어느 쪽인지 알 수 없어 모두 제외
    top = m.groupby("hotelrec_hotel_id").ratio.transform("max")
    m = m[m.ratio == top]
    n_amb = int(m.hotelrec_hotel_id.duplicated(keep=False).sum())
    m = m[~m.hotelrec_hotel_id.duplicated(keep=False)]
    print(f"모호한 매칭 제외: Kaggle 호텔 {n_amb}개")
    print(f"매칭: Kaggle 호텔 {len(km):,}개 중 {len(m):,}개 (정확 {(m.match == 'exact').sum()}, "
          f"유사 {(m.match == 'fuzzy').sum()})")
    ex = m.merge(km[["hotel_id", "hotel_name", "city"]], left_on="kaggle_hotel_id", right_on="hotel_id") \
          .merge(hm[["hotel_id", "hotel_name", "location"]], left_on="hotelrec_hotel_id", right_on="hotel_id",
                 suffixes=("_kaggle", "_hotelrec"))
    ex = ex[["kaggle_hotel_id", "hotel_name_kaggle", "city", "hotelrec_hotel_id", "hotel_name_hotelrec",
             "location", "match", "ratio"]]
    save_table(ex.sort_values(["city", "hotel_name_kaggle"]), "06_matched_hotels")
    print("유사 매칭 예시 (검토용):")
    print(ex[ex.match == "fuzzy"].head(10)[["hotel_name_kaggle", "hotel_name_hotelrec", "ratio"]]
          .round(3).to_string(index=False))

    # (2) 신규 사용자 예측
    s = np.load(WORK_DIR / "hotel_scores.npz")
    A = np.load(WORK_DIR / "hotel_aspects.npy")
    w_g = np.load(WORK_DIR / "global_weights.npy")
    lam = float(s["hybrid_lambda"])
    cb = w_g[0] + A @ w_g[1:]
    preds = {
        "GlobalMean": np.full(len(A), float(s["base_mu"])),
        "Baseline(μ+b_i)": s["base_mu"] + s["base_bi"],
        "MF(μ+b_i)": s["mf_mu"] + s["mf_bi"],
        "Content(aspect, 전역 w)": cb,
        f"Hybrid(λ={lam})": lam * (s["mf_mu"] + s["mf_bi"]) + (1 - lam) * cb,
    }
    preds = {name: np.clip(p, 1, 5) for name, p in preds.items()}

    # (3) 평가
    to_hr = dict(zip(m.kaggle_hotel_id, m.hotelrec_hotel_id))
    kr = k[k.hotel_id.isin(to_hr)].copy()
    kr["hr_id"] = kr.hotel_id.map(to_hr)
    print(f"매칭된 호텔의 Kaggle 리뷰 {len(kr):,}건 / 전체 {len(k):,}건")
    hotel_mean = kr.groupby("hr_id").rating.mean()
    rows, type_rows = [], []
    for name, p in preds.items():
        pr = p[kr.hr_id]
        ph = p[hotel_mean.index]
        rows.append({"model": name, "review_RMSE": rmse(kr.rating, pr), "review_MAE": mae(kr.rating, pr),
                     # 상수 예측(GlobalMean)은 순위가 없어 상관계수가 정의되지 않음 (NaN)
                     "review_spearman": spearmanr(pr, kr.rating).statistic if pr.std() > 0 else np.nan,
                     "hotel_pearson": pearsonr(ph, hotel_mean).statistic if ph.std() > 0 else np.nan,
                     "hotel_spearman": spearmanr(ph, hotel_mean).statistic if ph.std() > 0 else np.nan})
        if pr.std() > 0:
            row = {"model": name}
            for t, g in kr.groupby("traveler_type"):
                row[t] = spearmanr(p[g.hr_id], g.rating).statistic
            type_rows.append(row)
    res = pd.DataFrame(rows)
    print(f"평균 점수: Kaggle {kr.rating.mean():.3f} / HotelRec 학습 μ {float(s['base_mu']):.3f}")
    print(res.round(4).to_string(index=False))
    save_table(res, "06_kaggle_matched_prediction")
    by_type = pd.DataFrame(type_rows)
    print(by_type.round(4).to_string(index=False))
    save_table(by_type, "06_kaggle_matched_by_traveler_type")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, name in zip(ax, ["MF(μ+b_i)", f"Hybrid(λ={lam})"]):
        ph = preds[name][hotel_mean.index]
        a.scatter(ph, hotel_mean, s=8, alpha=0.5)
        a.set_xlabel(f"HotelRec 예측 ({name})")
        a.set_ylabel("Kaggle 호텔 평균 점수 (5점)")
        a.set_title(f"r = {pearsonr(ph, hotel_mean).statistic:.3f}")
    plt.suptitle(f"매칭 호텔 {len(hotel_mean):,}개: HotelRec 신규 사용자 예측 vs Booking.com 실제")
    save_fig(plt, "06_kaggle_matched_prediction")


if __name__ == "__main__":
    main()
