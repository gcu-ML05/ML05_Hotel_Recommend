"""
7단계 (선택): Item-kNN 하이퍼파라미터 선택 (val)

기존판(사용자 평균 기준, adjusted cosine)과 개선판(KNNBaseline)을 val 분할에서 비교하고
06_train_evaluate.py의 기본값(k, shrink, damp)을 고른 근거를 남긴다.

    val RMSE              : 모든 사용자의 val 리뷰 평점 예측
    HR@10 (predict)       : 평점 예측값으로 순위를 매긴 경우
    HR@10 (rank_score)    : 순위용 점수(공동 방문 유사도 합)로 순위를 매긴 경우
    neighbor_coverage     : val 쌍 중 사용자가 평가한 이웃이 하나라도 있는 비율
    (Top-N은 val 사용자 5,000명, 정답 1 + 무작위 후보 99)

설정 하나당 유사도 계산에 약 4~5분 걸린다 (전체 약 30분).
출력: Pipeline/outputs/07_itemknn_tuning.csv
"""
import time

import numpy as np
import pandas as pd

from common import SEED, ensure_dirs, load_split, ranking_metrics, rmse, sample_candidates, save_table
from models import BiasBaseline, ItemKNN

# (center, shrink, k, damp 목록)
CONFIGS = [
    ("user_mean", 10, 50, [0]),                  # 기존판
    ("baseline", 100, 50, [0, 0.03, 0.3, 1]),
    ("baseline", 100, 200, [0.03]),
    ("baseline", 100, 500, [0.03]),
    ("baseline", 100, 2000, [0.03]),
]


def main():
    ensure_dirs()
    df, n_users, n_hotels = load_split()
    tr, va = df[df.split == "train"], df[df.split == "val"]
    u, i, r = tr.user_id.to_numpy(), tr.hotel_id.to_numpy(), tr.rating.to_numpy(float)
    vu, vi, vr = va.user_id.to_numpy(), va.hotel_id.to_numpy(), va.rating.to_numpy(float)

    rng = np.random.default_rng(SEED)
    users = np.sort(rng.choice(va.user_id.unique(), 5000, replace=False))
    pos = va.set_index("user_id").loc[users, "hotel_id"].to_numpy()
    cand = sample_candidates(df, users, pos, n_hotels, 99, rng)
    cu, ci = np.repeat(users, cand.shape[1]), cand.ravel()

    base = BiasBaseline().fit(u, i, r, n_users, n_hotels)
    rows = [{"model": "Baseline(μ+b_u+b_i)", "val_RMSE": rmse(vr, base.predict(vu, vi)),
             "HR@10_predict": ranking_metrics(base.predict(cu, ci).reshape(cand.shape))[0]["HR@10"]}]
    for center, shrink, k, damps in CONFIGS:
        t0 = time.time()
        m = ItemKNN(k=k, shrink=shrink, center=center).fit(u, i, r, n_users, n_hotels)
        den = np.asarray((m.B[vu] @ m.ST)[np.arange(len(vu)), vi]).ravel()
        hr_rank = ranking_metrics(m.rank_score(cu, ci).reshape(cand.shape))[0]["HR@10"]
        for d in damps:
            m.damp = d
            rows.append({"model": f"ItemKNN center={center} shrink={shrink} k={k} damp={d}",
                         "val_RMSE": rmse(vr, m.predict(vu, vi)),
                         "HR@10_predict": ranking_metrics(m.predict(cu, ci).reshape(cand.shape))[0]["HR@10"],
                         "HR@10_rank_score": hr_rank, "neighbor_coverage": float((den > 0).mean()),
                         "fit_s": round(time.time() - t0)})
            print(rows[-1], flush=True)
    tab = pd.DataFrame(rows)
    print(tab.round(4).to_string(index=False))
    save_table(tab, "07_itemknn_tuning")


if __name__ == "__main__":
    main()
