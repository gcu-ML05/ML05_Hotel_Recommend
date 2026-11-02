"""
9단계: Kaggle aspect 키워드 사전(LEXICON) 점검

08_kaggle_generalization.py의 결과(여행 유형별 aspect 언급 비율, 페르소나 대응)는 수작업 키워드 목록에 좌우된다.
이 스크립트는 사전을 고치기 전에 근거를 만드는 용도이다.

(1) 키워드별 빈도: 긍정/부정 리뷰에서 각 키워드가 나온 리뷰 비율. 한 단어가 aspect 언급의 대부분을 차지하면
    (예: rooms의 "room") 결과가 그 단어 하나에 끌려간다.
(2) aspect 간 동시 언급: 같은 리뷰에서 두 aspect가 함께 잡히는 비율. "bed"(rooms)와 "mattress"(sleep_quality)처럼
    경계가 겹치는 aspect를 찾는다.
(3) 민감도: 키워드를 하나씩 뺐을 때 여행 유형별 언급 편차(%p)가 얼마나 바뀌는지.
    많이 바뀌는 키워드가 결과를 좌우하는 키워드이므로 팀 검토 우선순위가 된다.

실행 위치: 팀 저장소 Pipeline/ 폴더에 복사한 뒤  cd Pipeline && python 09_lexicon_audit.py
출력: Pipeline/outputs/07_keyword_freq.csv, 07_aspect_cooccurrence.csv, 07_keyword_sensitivity.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common import ASPECTS, KAGGLE_DIR, SEED, ensure_dirs, save_table

# 08_kaggle_generalization.py의 LEXICON을 그대로 가져온다 (사전을 한 곳에서만 관리하기 위해)
# 파일 이름이 숫자로 시작해 일반 import가 안 되므로 importlib로 읽는다
# spec_from_file_location(모듈 이름, 파일 경로): 파일에서 모듈 명세 생성 / module_from_spec: 빈 모듈 생성
# spec.loader.exec_module(모듈): 파일 코드를 실행해 모듈을 채움 (08의 main()은 __name__ 검사로 실행되지 않음)
_spec = importlib.util.spec_from_file_location("kaggle08", Path(__file__).parent / "08_kaggle_generalization.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LEXICON = _mod.LEXICON

SAMPLE = 100_000      # 민감도 계산은 표본으로 (키워드 수 x 재계산이라 전체는 느림)


def contains(texts, words):
    """texts 중 words의 단어가 하나라도 (단어 경계로) 들어 있는 리뷰 여부"""
    import re
    # re.escape(문자열): 정규식 특수문자를 이스케이프, \b: 단어 경계 (room이 bathroom 안에서 잡히지 않도록)
    pat = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
    # Series.str.contains(정규식, regex=True): 각 텍스트에 정규식이 있는지 불리언으로 반환
    return texts.str.contains(pat, regex=True).to_numpy()


def trip_type_dev(pos, neg, lex, groups):
    """08과 같은 방식으로 여행 유형별 aspect 언급 비율의 전체 평균 대비 편차(%p)"""
    M = np.column_stack([contains(pos, lex[a]) for a in ASPECTS]).astype(float) \
        + np.column_stack([contains(neg, lex[a]) for a in ASPECTS]).astype(float)
    has = M.sum(1) > 0
    share = M[has] / M[has].sum(1, keepdims=True)
    S = pd.DataFrame(share, columns=ASPECTS)
    g = S.groupby(groups[has].to_numpy()).mean()
    return (g - share.mean(0)) * 100


def main():
    """(1) 키워드 빈도 → (2) aspect 동시 언급 → (3) 키워드 제거 민감도 순서로 계산하고 표를 저장한다."""
    ensure_dirs()
    k = pd.read_csv(KAGGLE_DIR / "kaggle_test.csv", encoding="utf-8-sig",
                    usecols=["traveler_type", "positive_review", "negative_review"])
    pos = k.positive_review.fillna("").str.lower()
    neg = k.negative_review.fillna("").str.lower()

    # (1) 키워드별 빈도
    rows = []
    for a, words in LEXICON.items():
        hit_a = contains(pos, words) | contains(neg, words)
        for w in words:
            hp, hn = contains(pos, [w]), contains(neg, [w])
            rows.append({"aspect": a, "keyword": w, "pos_%": 100 * hp.mean(), "neg_%": 100 * hn.mean(),
                         # 이 aspect가 잡힌 리뷰 중 이 키워드가 들어 있는 비율 (높으면 aspect가 이 단어에 의존)
                         "share_of_aspect_hits_%": 100 * (hp | hn)[hit_a].mean() if hit_a.any() else 0})
    freq = pd.DataFrame(rows).sort_values(["aspect", "share_of_aspect_hits_%"], ascending=[True, False])
    print(freq.round(2).to_string(index=False))
    save_table(freq.round(3), "09_keyword_freq")

    # (2) aspect 간 동시 언급 (자카드 유사도)
    H = np.column_stack([contains(pos, LEXICON[a]) | contains(neg, LEXICON[a]) for a in ASPECTS])
    J = np.zeros((len(ASPECTS), len(ASPECTS)))
    for i in range(len(ASPECTS)):
        for j in range(len(ASPECTS)):
            J[i, j] = (H[:, i] & H[:, j]).sum() / max((H[:, i] | H[:, j]).sum(), 1)
    co = pd.DataFrame(J, index=ASPECTS, columns=ASPECTS)
    print(co.round(3).to_string())
    save_table(co.round(4), "09_aspect_cooccurrence", index=True)

    # (3) 민감도: 키워드 하나를 뺐을 때 여행 유형별 편차가 바뀌는 최대 폭(%p)
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
