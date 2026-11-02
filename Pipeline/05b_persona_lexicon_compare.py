"""
5b단계 (B 파트): 새 페르소나와 수정 키워드 사전을 05의 Kaggle 일반화 검증에 연결해 비교

05_kaggle_generalization.py는 (a) 02의 기존 페르소나, (b) 수작업 LEXICON을 쓴다.
여기서는 다음 네 가지를 같은 지표로 비교한다.

(1) 페르소나: 기존(02, 회귀 기반) vs 편차 기반(08)
(2) 배정 텍스트: 긍정+부정 언급 전체 vs 부정 언급만 (부정 리뷰의 불만이 '까다로움'과 더 직접 대응)
(3) 키워드 사전: 원본 LEXICON vs 수정 LEXICON (07 점검 결과 반영)
(4) 사전 강건성: 여행 유형별 aspect 언급 편차의 결론이 사전을 바꿔도 유지되는지

지표: Cramér's V (여행자 유형 x 페르소나 연관 강도), 스피어만 상관 (콘텐츠 점수와 리뷰어 점수)

선행 단계: 01~03, 05(persona_centroids.csv 사용), 08(08_persona_dev_share.csv 사용)
출력: Pipeline/outputs/05b_*.csv
"""
import importlib

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from common import ASPECTS, KAGGLE_DIR, OUT_DIR, WORK_DIR, ensure_dirs, save_table

# 파일명이 숫자로 시작해 import 문을 쓸 수 없으므로 importlib.import_module(모듈명)으로 불러옴 (원본 LEXICON 재사용)
base_mod = importlib.import_module("05_kaggle_generalization")
LEXICON_ORIG = base_mod.LEXICON

# 수정 사전: 07 점검에서 지적된 모호한 키워드 제거 + 빠진 키워드 추가
REMOVE = {
    "rooms": ["bed", "beds"],          # "bed was comfortable"은 sleep_quality와 겹침
    "cleanliness": ["smell"],          # "smell of smoke"와 "smells good breakfast" 구분 불가 (원본에는 smelly만 남김)
    "value": ["cheap"],                # "cheap furniture"는 rooms 품질 불만
}
ADD = {
    "service": ["waiter", "waitress", "manager", "housekeeping"],
    "cleanliness": ["hair", "filthy", "unclean"],
    "value": ["overcharged", "affordable", "reasonable", "bargain"],
    "location": ["restaurants", "shops", "attractions", "close to"],
    "sleep_quality": ["comfortable bed", "uncomfortable", "thin walls", "traffic", "bedding", "snoring"],
}


def build_patterns(lexicon):
    """키워드 사전 -> aspect별 정규식 (단어 경계 매칭). 05와 같은 방식."""
    import re
    return {a: r"\b(?:" + "|".join(re.escape(w) for w in ws) + r")\b" for a, ws in lexicon.items()}


def revised_lexicon():
    """원본 사전에서 REMOVE를 빼고 ADD를 더한 수정 사전 (중복 제거, 순서 유지)."""
    lex = {}
    for a, ws in LEXICON_ORIG.items():
        kept = [w for w in ws if w not in REMOVE.get(a, [])]
        lex[a] = list(dict.fromkeys(kept + ADD.get(a, [])))
    return lex


def mentions(pos_txt, neg_txt, lexicon):
    """리뷰별 aspect 언급 여부 행렬 (긍정 POS, 부정 NEG; 0/1, 리뷰 x 6 aspect)."""
    pat = build_patterns(lexicon)
    # Series.str.contains(정규식): 정규식이 텍스트에 포함되는지 불리언으로 반환
    pos = np.column_stack([pos_txt.str.contains(pat[a], regex=True) for a in ASPECTS]).astype(float)
    neg = np.column_stack([neg_txt.str.contains(pat[a], regex=True) for a in ASPECTS]).astype(float)
    return pos, neg


def unit(x):
    """행 단위 정규화 (코사인 유사도 계산용)."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_personas():
    """페르소나 두 벌 로드: (중심점 표, 기준 비율). 기준 비율은 코사인 비교에서 뺄 '평균 프로필'."""
    old = pd.read_csv(WORK_DIR / "persona_centroids.csv")
    users = pd.read_csv(WORK_DIR / "user_personas.csv")
    base_old = users[users.n_full >= 3][ASPECTS].mean().to_numpy()
    new = pd.read_csv(OUT_DIR / "08_persona_dev_share.csv")
    cnt = pd.read_csv(OUT_DIR / "08_persona_dev_centroids.csv").set_index("name").loc[new["name"], "n_users_fit"]
    # 새 페르소나의 기준 = 학습 사용자 수로 가중한 중심 평균
    base_new = (new[ASPECTS].to_numpy() * cnt.to_numpy()[:, None]).sum(0) / cnt.sum()
    out = {"기존(회귀 기반)": (old, base_old), "편차 기반(08)": (new, base_new)}
    # 08b(축소 강도·K 개선) 결과가 있으면 함께 비교
    tuned_path = OUT_DIR / "08b_persona_dev_share.csv"
    if tuned_path.exists():
        tuned = pd.read_csv(tuned_path)
        w = tuned["n_users_fit"].to_numpy()[:, None]
        out["편차 기반 개선(08b)"] = (tuned, (tuned[ASPECTS].to_numpy() * w).sum(0) / w.sum())
    return out


def evaluate(POS, NEG, meta, cent, base, mode, w_g):
    """한 조합(페르소나 x 배정 텍스트)의 지표 계산.

    배정: 리뷰 언급 비율 편차 R와 중심 편차 C = 중심 - 기준의 코사인 유사도가 최대인 페르소나.
    콘텐츠 점수: 호텔 aspect 평판(leave-one-out)에 가중치를 곱해 합산, 리뷰어 평점과 스피어만 상관.
    """
    M = (POS + NEG) if mode == "전체 텍스트" else NEG
    has = M.sum(1) > 0
    share = M[has] / M[has].sum(1, keepdims=True)
    R = share - share.mean(0)
    C = cent[ASPECTS].to_numpy() - base
    pidx = (unit(R) @ unit(C).T).argmax(1)
    trav = meta["traveler_type"].to_numpy()[has]
    ct = pd.crosstab(trav, cent["name"].to_numpy()[pidx])
    chi2, p, _, _ = chi2_contingency(ct)       # 카이제곱 독립성 검정: (통계량, p값, 자유도, 기대도수)
    v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))

    s_r = POS - NEG                             # 리뷰별 aspect 평판 (-1, 0, 1)
    hid = meta["hotel_id"].to_numpy()
    n_h = np.bincount(hid)                      # np.bincount: 호텔별 리뷰 수
    S_h = np.column_stack([np.bincount(hid, s_r[:, j]) for j in range(len(ASPECTS))])
    loo = ((S_h[hid] - s_r) / np.maximum(n_h[hid] - 1, 1)[:, None])[has]   # 자기 리뷰를 뺀 호텔 평판
    y = meta["rating"].to_numpy()[has]
    w_p = cent[ASPECTS].to_numpy()[pidx]
    # spearmanr(x, y): 순위 상관계수 (단조 관계의 강도, -1~1). .statistic이 상관계수
    rho = {"동일 가중": spearmanr(loo.mean(1), y).statistic,
           "전역 가중치": spearmanr(loo @ (w_g / w_g.sum()), y).statistic,
           "페르소나 가중치": spearmanr((loo * w_p).sum(1), y).statistic}
    sizes = pd.Series(cent["name"].to_numpy()[pidx]).value_counts(normalize=True)
    return {"n_reviews": int(has.sum()), "chi2": chi2, "p": p, "cramers_v": v,
            "rho_equal": rho["동일 가중"], "rho_global": rho["전역 가중치"], "rho_persona": rho["페르소나 가중치"],
            "largest_persona_share": sizes.max()}


def trip_deviation(POS, NEG, meta):
    """여행 유형별 aspect 언급 비율 - 전체 평균 (%p). 05와 같은 정의, 행 = traveler_type/trip_type 그룹."""
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
    """사전 2종 x 페르소나 x 배정 텍스트 조합을 평가하고, 사전 강건성과 수정 목록을 저장한다."""
    ensure_dirs()
    k = pd.read_csv(KAGGLE_DIR / "kaggle_test.csv", encoding="utf-8-sig",
                    usecols=["hotel_id", "rating", "trip_type", "traveler_type", "positive_review", "negative_review"])
    pos_txt = k.positive_review.fillna("").str.lower()
    neg_txt = k.negative_review.fillna("").str.lower()
    w_g = np.load(WORK_DIR / "global_weights.npy")[1:]
    personas = load_personas()
    lexicons = {"원본 사전": LEXICON_ORIG, "수정 사전": revised_lexicon()}

    rows, devs = [], {}
    for lname, lex in lexicons.items():
        POS, NEG = mentions(pos_txt, neg_txt, lex)
        devs[lname] = trip_deviation(POS, NEG, k)
        for pname, (cent, base) in personas.items():
            for mode in ["전체 텍스트", "부정 텍스트만"]:
                rows.append({"lexicon": lname, "persona": pname, "assign_by": mode,
                             **evaluate(POS, NEG, k, cent, base, mode, w_g)})
    comp = pd.DataFrame(rows)
    print(comp.round(4).to_string(index=False))
    save_table(comp.round(4), "05b_persona_comparison")

    # (4) 사전 강건성: 그룹별 편차 표가 사전 수정 후에도 같은지
    a, b = devs["원본 사전"], devs["수정 사전"]
    A, B = a[ASPECTS], b[ASPECTS]
    same_top = (A.idxmax(axis=1) == B.idxmax(axis=1))
    same_bottom = (A.idxmin(axis=1) == B.idxmin(axis=1))
    rob = pd.DataFrame({"column": a["column"], "top_orig": A.idxmax(axis=1), "top_rev": B.idxmax(axis=1),
                        "top_same": same_top, "bottom_orig": A.idxmin(axis=1), "bottom_rev": B.idxmin(axis=1),
                        "bottom_same": same_bottom, "max_abs_diff_pp": (A - B).abs().max(axis=1)})
    print(rob.round(2).to_string())
    flat = np.corrcoef(A.to_numpy().ravel(), B.to_numpy().ravel())[0, 1]
    print(f"그룹 x aspect 편차 표의 상관(원본 vs 수정): {flat:.3f}, 최상위 aspect 유지 {same_top.mean():.0%}, "
          f"최하위 유지 {same_bottom.mean():.0%}")
    save_table(rob.round(3), "05b_lexicon_robustness", index=True)
    save_table(B.assign(column=b["column"]).round(3), "05b_aspect_mention_by_trip_type_revised", index=True)

    # 수정 사전 목록 기록 (팀 검토용)
    changes = [{"aspect": x, "action": "remove", "keyword": w} for x, ws in REMOVE.items() for w in ws]
    changes += [{"aspect": x, "action": "add", "keyword": w} for x, ws in ADD.items() for w in ws]
    save_table(pd.DataFrame(changes), "05b_lexicon_changes")


if __name__ == "__main__":
    main()
