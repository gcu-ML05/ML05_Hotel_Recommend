"""
5단계: Kaggle(Booking.com) 데이터로 일반화 검증

Kaggle 데이터에는 사용자 ID와 세부 평점이 없으므로 제안서 2절대로 리뷰 1건을 신규 사용자 1명으로 보고,
리뷰 본문에서 언급된 aspect로 선호를 추정한다.

(1) aspect 언급 추출
    aspect별 키워드 사전(LEXICON, 팀이 정한 수작업 목록)으로 긍정/부정 리뷰에서 각 aspect가 언급됐는지 센다.
    리뷰의 aspect 언급 비율 = 언급 수 / 전체 언급 수  (언급이 하나도 없는 리뷰는 제외)

(2) 여행 유형별 aspect 언급 비교
    traveler_type(Couple, Business 등), trip_type별 언급 비율이 전체 평균과 얼마나 다른지 (퍼센트포인트)

(3) HotelRec 페르소나와의 대응
    리뷰의 언급 비율 편차(전체 평균 대비)와 페르소나 중심점의 편차(HotelRec 전체 평균 대비)의 코사인 유사도가
    가장 큰 페르소나를 그 리뷰에 배정하고, 여행자 유형 x 페르소나 교차표의 독립성을 카이제곱 검정한다.

(4) 콘텐츠 기반 점수의 일반화
    호텔별 aspect 평판 = 그 호텔 다른 리뷰들의 (긍정 언급 여부 - 부정 언급 여부) 평균 (자기 리뷰 제외, leave-one-out)
    호텔 점수 = Σ_k 가중치_k x 평판_k 가 리뷰어의 실제 점수와 얼마나 상관있는지 (스피어만 순위상관)
    가중치: 동일 가중 / HotelRec 전역 가중치 / 리뷰에 배정된 페르소나 가중치

출력: Pipeline/outputs/05_*.csv, figures/05_*.png
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from common import ASPECTS, KAGGLE_DIR, WORK_DIR, ensure_dirs, save_fig, save_table, setup_plot

# aspect별 키워드 (소문자, 단어 경계로 매칭). bed는 rooms, mattress/pillow는 sleep_quality로 구분
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
    print(f"Kaggle 리뷰 {len(k):,}건")

    # (1) aspect 언급 (긍정/부정 각각: 언급 여부 0/1)
    pos_txt = k.positive_review.fillna("").str.lower()
    neg_txt = k.negative_review.fillna("").str.lower()
    POS = np.column_stack([pos_txt.str.contains(PATTERNS[a], regex=True) for a in ASPECTS]).astype(float)
    NEG = np.column_stack([neg_txt.str.contains(PATTERNS[a], regex=True) for a in ASPECTS]).astype(float)
    M = POS + NEG
    has = M.sum(1) > 0
    share = M[has] / M[has].sum(1, keepdims=True)
    kk = k[has].reset_index(drop=True)
    print(f"aspect 언급이 있는 리뷰 {has.sum():,}건 ({has.mean():.1%})")
    overall = share.mean(0)
    print("전체 언급 비율:", dict(zip(ASPECTS, overall.round(3))))

    # (2) 여행 유형별 언급 비율 (전체 평균 대비 %p)
    S = pd.DataFrame(share, columns=ASPECTS)
    tabs = []
    for col in ["traveler_type", "trip_type"]:
        g = S.groupby(kk[col]).mean()
        dev = (g - overall) * 100
        dev.insert(0, "n_reviews", kk[col].value_counts().reindex(dev.index))
        tabs.append(dev.rename_axis("group").reset_index().assign(column=col))
    dev_tab = pd.concat(tabs, ignore_index=True)
    print(dev_tab.round(2).to_string(index=False))
    save_table(dev_tab.round(3), "05_aspect_mention_by_trip_type")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    D = dev_tab.set_index("group")[ASPECTS]
    lim = np.abs(D.to_numpy()).max()
    im = ax.imshow(D.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(ASPECTS)), ASPECTS, rotation=30)
    ax.set_yticks(range(len(D)), D.index)
    for a in range(D.shape[0]):
        for b in range(D.shape[1]):
            ax.text(b, a, f"{D.iat[a, b]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("여행 유형별 aspect 언급 비율 - 전체 평균 (%p)")
    plt.colorbar(im)
    save_fig(plt, "05_aspect_mention_by_trip_type")

    # (3) 페르소나 대응
    cent = pd.read_csv(WORK_DIR / "persona_centroids.csv")
    users = pd.read_csv(WORK_DIR / "user_personas.csv")
    base = users[users.n_full >= 3][ASPECTS].mean().to_numpy()       # 페르소나 학습 사용자의 평균 비율
    C = cent[ASPECTS].to_numpy() - base
    R = share - overall

    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    kk["persona"] = cent["name"].to_numpy()[(unit(R) @ unit(C).T).argmax(1)]
    ct = pd.crosstab(kk.traveler_type, kk.persona)
    # chi2_contingency(교차표): 두 범주형 변수가 독립인지 검정. 반환값 (카이제곱 통계량, p값, 자유도, 기대도수)
    chi2, p, dof, _ = chi2_contingency(ct)
    cramer_v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))   # 0~1, 연관 강도
    print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1).to_string())
    print(f"카이제곱 {chi2:.1f}, 자유도 {dof}, p={p:.3g}, Cramér's V = {cramer_v:.3f}")
    save_table((ct.div(ct.sum(axis=1), axis=0) * 100).round(2), "05_traveler_type_x_persona", index=True)

    # 여행자 유형 전체의 언급 프로필이 어느 페르소나와 가장 비슷한지
    G = S.groupby(kk.traveler_type).mean().to_numpy() - overall
    sim = pd.DataFrame(unit(G) @ unit(C).T, index=S.groupby(kk.traveler_type).mean().index, columns=cent["name"])
    print(sim.round(3).to_string())
    save_table(sim.round(4), "05_traveler_type_persona_similarity", index=True)

    # (4) 콘텐츠 기반 점수의 일반화: 호텔 aspect 평판(leave-one-out)과 리뷰어 점수의 순위상관
    s_r = POS - NEG                                                   # 리뷰별 aspect 평판 (-1, 0, 1)
    hid = k.hotel_id.to_numpy()
    n_h = np.bincount(hid)
    S_h = np.column_stack([np.bincount(hid, s_r[:, j]) for j in range(len(ASPECTS))])
    loo = (S_h[hid] - s_r) / np.maximum(n_h[hid] - 1, 1)[:, None]    # 자기 리뷰를 뺀 호텔 평판
    loo = loo[has]
    w_g = np.load(WORK_DIR / "global_weights.npy")[1:]
    w_p = cent.set_index("name").loc[kk.persona, ASPECTS].to_numpy()
    scores = {"동일 가중": loo.mean(1), "HotelRec 전역 가중치": loo @ (w_g / w_g.sum()),
              "배정된 페르소나 가중치": (loo * w_p).sum(1)}
    rows = []
    for name, sc in scores.items():
        row = {"weights": name, "all": spearmanr(sc, kk.rating).statistic}   # 스피어만 순위상관계수
        for t, g in kk.groupby("traveler_type"):
            row[t] = spearmanr(sc[g.index], g.rating).statistic
        rows.append(row)
    corr = pd.DataFrame(rows)
    print(corr.round(4).to_string(index=False))
    save_table(corr.round(4), "05_content_score_generalization")


if __name__ == "__main__":
    main()
